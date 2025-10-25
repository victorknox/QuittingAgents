import json

class RobustJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if hasattr(obj, "as_dict") and callable(obj.as_dict):
            return obj.as_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj) 