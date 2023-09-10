import json

# OLD FUNCTION
# import sys
# def handler(event, context):
#     return 'Hello from AWS Lambda using Python' + sys.version + '!'

# COPY OF THE NEW FUNCTION
# The actual function is inside lambda_function.py

# def handler(event, context):

#     # for logging purposes
#     # check this in CloudWatch
#     print(event)
#     print(type(event))

#     # generate message
#     results = event
#     message = 'Hello {} {}!'.format(results['first_name'], results['last_name'])  

#     # standard return function, note json.dumps(message) to remove problematic format like
#     return {
#         'statusCode': 200,
#         'headers': {'Content-Type': 'application/json'},
#         'body': json.dumps(message)
#     }

# # for lab exercise #1
# def handler(event, context):
#     return "I love MLOps"

# for lab exercise #2
def handler(event,context):
    return context[0] + context[1]