
from .._dependencies import _try_optional_dependencies_gui
from .screens.preprocessor import Preprocessor
from .screens.loader import load_preprocessor
from .screens.loader import load_classifier
from .screens.loader import load_regressor


def _optional_dependency_placeholder(*args, **kwargs):
    raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[gui]' to use "
                          "this feature.")

class OptionalDependencyPlaceholder:

    def __init__(self):
        raise ImportError("Optional dependencies not found. Please install "
                          "them with `pip install xplainable[gui]' to use "
                          "this feature.")

# import functions and classes as normal if optional dependencies are found
if _try_optional_dependencies_gui():
    from .gradio.fine_tuning import XFineTuner
    from .screens.classifier import classifier
    from .screens.regressor import regressor
    from .screens.evaluate import EvaluateClassifier, EvaluateRegressor
    from .screens.save import ModelPersist
    from .screens.scenario import ScenarioClassification, ScenarioRegression

# otherwise, import placeholder functions and classes
else:
    classifier = _optional_dependency_placeholder
    regressor = _optional_dependency_placeholder

    EvaluateClassifier = OptionalDependencyPlaceholder
    EvaluateRegressor = OptionalDependencyPlaceholder
    Preprocessor = OptionalDependencyPlaceholder
    ModelPersist = OptionalDependencyPlaceholder
    ScenarioClassification = OptionalDependencyPlaceholder
    ScenarioRegression = OptionalDependencyPlaceholder
    XFineTuner = OptionalDependencyPlaceholder
