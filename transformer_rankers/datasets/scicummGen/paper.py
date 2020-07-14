from os import listdir
from os.path import join

from .annotation import Annotation
from .document import Document


class Paper:

    def __init__(self, path):
        reference_path = join(path, 'Reference_XML')
        annotations_path = join(path, 'annotation')

        self.reference = Document(join(reference_path, listdir(reference_path)[0]))
        self.annotation = Annotation(join(annotations_path, listdir(annotations_path)[0]))


