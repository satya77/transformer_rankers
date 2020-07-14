from bs4 import BeautifulSoup

class Annotation:

    def __init__(self, path):
        self.citances = []

        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                if '|' in line:
                    self._process_line(line)

    def _process_line(self, line):
        citance = {}
        fields = line.split('|')

        for raw_field in fields:
            field = raw_field.strip()

            if field.startswith('Reference Article'):
                citance['RP'] = field.split(':')[1].strip()
            if field.startswith('Citing Article'):
                citance['CP'] = field.split(':')[1].split('.')[0].strip()
            if field.startswith('Citation Offset'):
                citance['CO'] = self._get_int_list(field.split(':')[1].strip())
            if field.startswith('Reference Offset'):
                citance['RO'] = self._get_int_list(field.split(':')[1].strip())
            if field.startswith('Citation Text'):
                citance['query'] =BeautifulSoup((field.split(':')[1].strip()), "html.parser").s.string
            if field.startswith('Reference Text'):
                citance['passage'] =BeautifulSoup( (field.split(':')[1].strip()), "html.parser").s.string


        self.citances.append(citance)

    @staticmethod
    def _get_int_list(text):
        int_list = []
        text = text.strip('[]')
        for number in text.split(','):
            int_list.append(int(number.strip(' ').strip('\'')))
        return int_list
