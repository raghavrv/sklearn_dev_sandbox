from sklearn.utils import check_X_y, check_random_state

def drop_values(X, y=None, missing_mask=None,
                missing_values="NaN",
                missing_fraction=0.1,
                label_correlation=1.0,
                n_labels=1,
                labels=None,
                missing_mask_only=False,
                return_labels=False,
                return_missing_mask=False,
                copy=False,
                random_state=None):
    """Drop values based on a preset strategy.
    
    Attributes
    ----------
    
    X : ndarray like of shape (n_features, n_samples)
    
        Data, in which the values must be dropped and set to
        missing_values.
        
    y : ndarray like of shape (n_samples,), optional
        Target, in a supervised classification task.
        
    missing_mask : bool ndarray shape (n_features, n_samples), optional
        This is used to either denote the missing values that already
        exist in the data or simply to specify a missing mask to modify
        inplace.
        
    missing_values : {"NaN" (or np.nan) | int}, default "NaN"
        The missing value to use
        
    label_correlation : float, default 1.0
        1 (MNAR) - Randomly choose n_targets (or take from given
        targets) and correlate the missing values with the occurence
        of those target labels.
        
        0 (MCAR) - Randomly drop values without correlating the
        missingness with any target.
        
        Any value inbetween would constitute a noisy MNAR missingness.
        
    n_labels : int, optional, default 1
        The number of labels to pick at random and correlate with
        
    labels : 1D list/ndarray, optional, default None
        The list of labels (must match with the labels in y) to
        correlate with.
        
        If this is specified n_labels argument is ignored.
        
    missing_mask_only : bool, default False
        Whether to modify/return only the missing mask without
        touching the actual data X.
    
    return_missing_mask : bool, default False
        Whether to return the missing mask along with the data (X, y)
    
    return_labels : bool, default False
        Whether to return the picked labels
    
    copy : bool, default False
        Whether to copy the data (and missing_mask) or work inplace.
        
    random_state : int, optional
        The seed for the numpy's random number generator.
    Returns
    -------
    
    X, y (, missing_mask, labels) : Tuple
        Returns missing_mask if return_missing_mask is set to True
        Returns labels if return_labels is set to True
        
    """
    X, y = check_X_y(X, y)
    
    if missing_mask_only and not return_missing_mask:
        raise ValueError("Both missing_mask_only and return_missing_mask"
                         "cannot be True")
        
    if missing_fraction >= 1:
        raise ValueError("The missing_fraction cannot be greater than"
                         " or equal to 1.")
    
    if copy:
        X = X.copy()
        if missing_mask is not None:
            missing_mask = missing_mask.copy()
            
    if (isinstance(missing_values, str) and
            missing_values.lower() == "NaN"):
        missing_values = np.nan
        
    if missing_mask is None:
        if np.isnan(missing_values):
            missing_mask = np.isnan(X)
        else:
            missing_mask = X == missing_values
    
    n_samples, n_features = X.shape
    n_elements = n_samples * n_features
    
    current_n_missing = np.count_nonzero(missing_mask)
    required_n_missing = int(missing_fraction * n_elements)
    
    #print current_n_missing, required_n_missing
    
    if current_n_missing > required_n_missing:
        raise ValueError("There are currently %d missing values, "
                         "which is >= a fraction of %0.2f that is"
                         "expected to be missing."
                         % (current_n_missing, missing_fraction))
    
    rng = check_random_state(random_state)
    n_more_missing = required_n_missing - current_n_missing
    
    unique_labels = np.unique(y)
    n_unique_labels = len(unique_labels)
    
    if labels is None:
        # Labels is an int specifying the no of labels to correlate
        # with
        
        if n_labels > n_unique_labels:
            raise ValueError("The n_labels (%d) is greater than"
                             " no of unique labels in y (%d)"
                             % (n_labels, n_unique_labels))
        
        labels = rng.choice(n_unique_labels, n_labels, replace=False)
        # Reset the RNG as we don't want this operation to affect
        # the random selection
        rng = check_random_state(random_state)

    
    # Filter based on labels
    n_correlated_missing = int(n_more_missing * label_correlation)
    n_non_correlated_missing = n_more_missing - n_correlated_missing
    
    label_mask = np.zeros(missing_mask.shape, dtype=bool)
    if label_correlation != 0:
        for label in labels:
            label_mask[y==label] = True        

    # The logic of MCAR/MNAR is implemented here
    inv_missing_mask = ~missing_mask
    corr_available = inv_missing_mask & label_mask
    uncorr_available = inv_missing_mask & ~label_mask
    
    n_corr_available = np.count_nonzero(corr_available)
    n_uncorr_available = np.count_nonzero(uncorr_available)
    
    n_available = n_corr_available + n_uncorr_available
    
    if n_available < n_more_missing:
        raise ValueError("There are only %d values available for "
                         "dropping. %d more are needed to reach"
                         " the missing_fraction of %0.2f"
                         % (n_available, n_more_missing,
                            missing_fraction))
    
    n_corr_chosen = int(n_more_missing * label_correlation)
    if n_corr_chosen == 0:
        corr_chosen = []
    else:
        corr_chosen = rng.choice(n_corr_available,
                                 n_corr_chosen,
                                 replace=False)
        
    n_uncorr_chosen = n_more_missing - n_corr_chosen
    if n_uncorr_chosen == 0:
        uncorr_chosen = []
    else:
        uncorr_chosen = rng.choice(n_uncorr_available,
                                   n_more_missing - n_corr_chosen,
                                   replace=False)
    
    print ("No of (additional) correlated/uncorrelated missing "
           "values - %d/%d" % (n_corr_chosen, n_uncorr_chosen))
    # print ("Indices of correlated/uncorrelated missing values - %d/%d"
    #        % (corr_chosen, uncorr_chosen))
    all_corr_indices = np.where(corr_available)
    all_uncorr_indices = np.where(uncorr_available)
    
    #print all_available_indices
    missing_indices_corr = (all_corr_indices[0][corr_chosen],
                            all_corr_indices[1][corr_chosen])
    missing_indices_uncorr = (all_uncorr_indices[0][uncorr_chosen],
                              all_uncorr_indices[1][uncorr_chosen])
    missing_mask[missing_indices_corr] = True
    missing_mask[missing_indices_uncorr] = True
    
    if not missing_mask_only:
        X[missing_indices_corr] = missing_values
        X[missing_indices_uncorr] = missing_values
    
    ret = [X, y]
    
    if return_missing_mask:
        ret.append(missing_mask)
    
    if return_labels:
        ret.append(labels)
    
    return ret
