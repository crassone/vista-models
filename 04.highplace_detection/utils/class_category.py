class_category = [
    'detection-miss',
    'person',
    'person in high place',
]
no_alarm_class_category = [
    'detection-miss',
    'person',
]

def classname2number(label_name):
    for idx, category in enumerate(class_category):
        if label_name == category:
            return idx
