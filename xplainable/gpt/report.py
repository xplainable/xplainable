import xplainable as xp

def gpt_explainer(
        model_id, version_id, target_info='', project_objective='',
        max_features=15
        ):
    """
    Generates a report for a given model and version.

    Args:
        model_id (str): The model ID.
        version_id (str): The version ID.
        target_info (str): The target information.
        project_objective (str): The project objective.
        max_features (int): The maximum number of features to analyse.

    Returns:
        str: The report.

    """

    if xp.client is None:
        raise ValueError(
            "You must initialise a valid API key to use this feature.") \
                from None

    report = xp.client._gpt_report(
        model_id, version_id, target_info, project_objective, max_features)
    
    return report
