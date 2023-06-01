""" Copyright Xplainable Pty Ltd, 2023"""

import pandas as pd

def _generate_explain_plot_data(model, partition):

    def get_plot_data(f, p):
        """ 
        Args:
            f (str): Feature
            p (str): Partition
        """
        _profile = model.partitions[p]['profile']
        if f in _profile['numeric']:
            prof = pd.DataFrame(
                {i: v for i, v in _profile['numeric'][f].items() if \
                    len(v) > 0}).T.reset_index()

            if prof.empty:
                return

            prof['value'] = prof['lower'].round(2).astype(str) + " - " + \
                prof['upper'].round(2).astype(str)

            prof = prof[['value', 'score']]
            
        elif f in _profile['categorical']:
            prof = pd.DataFrame({i: v for i, v in _profile['categorical'][
                f].items() if len(v) > 0}).T.reset_index()

            if prof.empty:
                return

            prof = prof[['categories', 'score']]
            prof['categories'] = prof['categories'].apply(lambda x: list(x))
            prof = prof.rename(columns={'categories': 'value'})
            prof = prof.explode('value')

        else:
            return

        prof['feature'] = f
        prof['score_label'] = prof['score'].apply(
            lambda x: str(round(x*100, 1)))

        return prof.reset_index()
    
    feat_imp = model.partitions[partition]["feature_importances"]
    fimp = pd.DataFrame(
        {i: {'importance': v} for i, v in feat_imp.items()}).T.reset_index()

    fimp = fimp.rename(columns={'index': 'feature'})
    fimp['importance_label'] = fimp['importance'].apply(
        lambda x: str(round(x*100, 1))+'%')

    plot_data = [get_plot_data(i, partition) for i in feat_imp.keys()]
    prof = pd.concat(
        [i for i in plot_data if i is not None]).reset_index(drop=True)
    
    return fimp, prof

def generate_partition_data(model):

    data = {p: {} for p in model.partitions}
    for p, v in data.items():
        fi, prof = _generate_explain_plot_data(model, p)
        data[p]['feature_importances'] = fi
        data[p]['profile'] = prof
        
    return data
