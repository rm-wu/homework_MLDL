from sklearn.model_selection._search import ParameterGrid, _check_param_grid, _check_multimetric_scoring
from sklearn.base import clone
from sklearn.model_selection._validation import _aggregate_score_dicts, _index_param_value, _score
from itertools import product
from scipy.stats import rankdata
from collections import defaultdict
from functools import partial
from sklearn.utils import _message_with_time
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import _num_samples
import warnings
import time
from joblib import Parallel, delayed
import numpy as np

class GridSearch():
    def __init__(self, estimator, param_grid, scoring=None, n_jobs=None, verbose=False, refit=True):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.param_grid = param_grid
        _check_param_grid(param_grid)



    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    def fit(self, X_train, y_train, X_val, y_val, **fit_params):
        estimator = self.estimator
        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'


        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch='2*n_jobs')  # default value

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=False,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=np.nan,
                                    verbose=self.verbose)
        results = {}

        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = list(candidate_params)

                if self.verbose:
                    print(f"Fitting {len(n_candidates)}")

                out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                       X_train, y_train, X_val, y_val,
                                                       parameters=parameters,
                                                       **fit_and_score_kwargs)
                               for parameters,
                               in product(candidate_params))
                if len(out) < 1:
                    raise ValueError('No fits were performed.')

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, 1, all_out)
                return results

            self._run_search(evaluate_candidates)

            # For multi-metric evaluation, store the best_index_, best_params_ and
            # best_score_ iff refit is one of the scorer names
            # In single metric evaluation, refit_metric is "score"
            if self.refit or not self.multimetric_:
                # If callable, refit is expected to return the index of the best
                # parameter set.
                if callable(self.refit):
                    self.best_index_ = self.refit(results)
                    if not isinstance(self.best_index_, (int, np.integer)):
                        raise TypeError('best_index_ returned is not an integer')
                    if (self.best_index_ < 0 or
                            self.best_index_ >= len(results["params"])):
                        raise IndexError('best_index_ index out of range')
                else:
                    self.best_index_ = results["rank_test_%s"
                                               % refit_metric].argmin()
                    self.best_score_ = results["mean_test_%s" % refit_metric][
                        self.best_index_]
                self.best_params_ = results["params"][self.best_index_]

            if self.refit:
                self.best_estimator_ = clone(base_estimator).set_params(
                    **self.best_params_)
                refit_start_time = time.time()
                if y_train is not None:
                    self.best_estimator_.fit(X_train, y_train, **fit_params)
                else:
                    self.best_estimator_.fit(X_train, **fit_params)
                refit_end_time = time.time()
                self.refit_time_ = refit_end_time - refit_start_time

            # Store the only scorer not as a dict for single metric evaluation
            self.scorer_ = scorers if self.multimetric_ else scorers['score']

            self.results_ = results
            self.n_splits_ = 1

            return self

    def _format_results(self, candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        (test_score_dicts, test_sample_counts, fit_time,
         score_time) = zip(*out)

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)


        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights= None)
        return results


def _fit_and_score(estimator, X_train, y_train, X_val, y_val, scorer, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_estimator=False,
                   error_score='raise-deprecating'):
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print(" %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    # fit_params = fit_params if fit_params is not None else {}
    # fit_params = {k: _index_param_value(X, v, train)
    #               for k, v in fit_params.items()}
    fit_params = {}

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif error_score == 'raise-deprecating':
            warnings.warn("From version 0.22, errors during fit will result "
                          "in a cross validation score of NaN by default. Use "
                          "error_score='raise' if you want an exception "
                          "raised or error_score=np.nan to adopt the "
                          "behavior from version 0.22.",
                          FutureWarning)
            raise

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(estimator, X_val, y_val, scorer, is_multimetric)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer,
                                  is_multimetric)
    if verbose > 2:
        if is_multimetric:
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += ("%.3f" % test_scores if not return_train_score else
                    "(train=%.3f, test=%.3f)" % (train_scores, test_scores))

    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time(msg, total_time))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_val))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret

