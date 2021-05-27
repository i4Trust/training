from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact


# @env(infer_pip_packages=True)
@env(
    pip_packages=[
        'scikit-learn==0.24.2',
        'numpy==1.20.2',
        'pytz==2021.1'
    ]
)
@artifacts([SklearnModelArtifact('flow_model')])
class TestFlowModel(BentoService):
    @api(input=JsonInput(), output=JsonOutput())
    def predict(self, rainfall):
        # transform the input as a numpy array
        flow = self.artifacts.flow_model.predict(rainfall)
        return(flow)
