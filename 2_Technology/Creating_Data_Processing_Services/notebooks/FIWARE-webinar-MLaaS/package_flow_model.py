import pickle
from flow_predictions import TestFlowModel

# Load the previously saved sklearn model
with open('./linear_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Create an instance of the FlowModel service
flow_predictions_service = TestFlowModel()

# Pack the trained model
flow_predictions_service.pack('flow_model', model)

# Save the prediction service to disk for model serving
saved_path = flow_predictions_service.save()

print('done')
