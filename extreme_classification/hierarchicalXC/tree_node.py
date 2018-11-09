

class TreeNode(object):

    def __init__(self, classifier, class_list):
        self.left = None
        self.right = None
        self.parent = None
        self.classifier = classifier
        self.class_list = class_list
