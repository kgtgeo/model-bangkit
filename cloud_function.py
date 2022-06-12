from googleapiclient import discovery
from google.api_core.client_options import ClientOptions
import json

def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    print('requesting to {}/{}'.format(api_endpoint, name))

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    print('request executed')

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


CLOUD_PROJECT = 'bangkit-352714'
MODEL = 'bangkit_0'
REGION = 'us-east1'
VERSION = 'v1'

FIELDS = ['age', 'heart', 'weight', 'temperature', 'height', 'duration']

def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    data = request.get_json()
    data = data if data else dict()

    features = []
    for field in FIELDS:
        features.append(data.get(field, 1))

    INPUT = [features]
    prediction = predict_json(CLOUD_PROJECT, REGION, MODEL, INPUT, 'v1')
    print("Prediction", INPUT, "with result", prediction)
    print("after prediction", prediction, "type of prediction", type(prediction), type(prediction[0]))
    prediction_value = prediction[0][0]
    print("prediction value", prediction_value)
    res = {}
    res['prediction'] = prediction_value
    print("res", res)
    return json.dumps(res)

    # if request.args and 'message' in request.args:
    #     return request.args.get('message')
    # elif request_json and 'message' in request_json:
    #     return request_json['message']
    # else:
    #     return f'Hello World!'
