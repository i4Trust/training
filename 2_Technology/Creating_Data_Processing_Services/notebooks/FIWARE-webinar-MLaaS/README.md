# MLaaS webinar

Contains two Jupyter notebooks which were used to run each step of the demo during the webinar:
* `Fiware-webinar` is the one that was demonstrated, i.e. the one that works together with a deployed BentoML model and a consumer application
* `Fiware-webinar-the-complete-requests` is a notebook that simulates all parts, i.e. a BentoML model and a consumer application. That is, it 
makes all requests describes into the sequence diagram, except obviously the notification sent by the Context Broker. One can use Postman 
and its Mock Servers feature to check if the context broker is sending notification.

Additional material:
* A pickle of the linear model created during the webinar (`linear_model.pickle`)
* Two python scripts to create a Bento for this model, which provides a simple example on how to create a packaged model with BentoML

Once created by running `python package_flow_model.py`, the ML model can simply be dockerised with:
```shell
save_path=$(bentoml get YourModelNamel:latest --print-location --quiet)
docker build -t name:tag $save_path
```
