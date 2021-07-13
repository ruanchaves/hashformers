import json
import string

def get_from_dict(dct, *keys):
    for key in keys:
        dct = dct[key]
    return dct

def get_category(char):
    if not char.strip():
        return "SPACE"
    elif char in string.punctuation:
        return "PUNCT"
    else:
        return "CHAR"

class DFA(object):

    def __init__(self, filename, initial_state='s0', chars=['B', 'M', 'E', 'S']):
        with open(filename,'r') as f:
            self.dfa = json.load(f)
        self.initial_state = initial_state
        self.chars = chars

    def run_DFA(self, string_input):
        state = self.initial_state
        output = ''
        if not any(string_input.endswith(x) for x in string.punctuation):
            string_input += '.'
        for char in string_input:
            category = get_category(char)
            state = get_from_dict(self.dfa, state, category)
            output += state
        output = [ x for x in output if x in self.chars]
        return output

def main():
    dfa = DFA('dfa.json')
    dfa.run_DFA('João e Maria.')

if __name__ == '__main__':
    main()

