from xml.dom import minidom
from xml.parsers.expat import ExpatError


class Document:

    def __init__(self, path):
        with open(path, 'rb') as file:
            raw_content = file.read()

        content = raw_content.decode('utf-8', 'ignore')

        try:
            xml = minidom.parseString(content)
            tags = xml.getElementsByTagName('S')

            sid_max = 0
            ssid_max = {}
            section_max = 0

            section = 0

            for s in tags:
                sid = s.attributes['sid'].value
                if sid is '':
                    continue

                try:
                    ssid = s.attributes['ssid'].value
                except KeyError:
                    continue

                if ssid is '1':
                    section += 1

                # Find max section
                if section_max < section:
                    section_max = section

                sid = int(sid)
                ssid = int(ssid)

                # Find max sid
                if sid_max < sid:
                    sid_max = sid

                # Find max ssid
                try:
                    if ssid_max[section] < ssid:
                        ssid_max[section] = ssid
                except KeyError:
                    ssid_max[section] = ssid

            section = 0
            self.sentences = {}

            for s in tags:
                sid = s.attributes['sid'].value
                if sid is '':
                    continue

                try:
                    ssid = s.attributes['ssid'].value
                except KeyError:
                    continue

                text = s.firstChild.nodeValue

                if ssid is '1':
                    section += 1

                sentence = {'sid': int(sid),
                            'sid_max': sid_max,
                            'ssid': int(ssid),
                            'ssid_max': ssid_max[section],
                            'section': section,
                            'section_max': section_max,
                            'text': text}

                self.sentences[int(sid)] = sentence

        except ExpatError as e:
            print('Unable to parse the file', path)
            # raise e

        except ValueError as e:
            print('Unable to parse the file', path)
            raise e
