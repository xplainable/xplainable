import xplainable as xp

def gpt_explainer(model_id, version_id, target_info='', other_details=''):

    if xp.client is None:
        raise ValueError(
            "You must initialise a valid API key to use this feature.") \
                from None

    report = xp.client._gpt_report(
        model_id, version_id, target_info, other_details)
    
    return report