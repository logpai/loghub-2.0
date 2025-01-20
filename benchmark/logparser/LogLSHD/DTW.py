import numpy as np
from fastdtw import fastdtw
import regex as re

class DTW_TemplateGenerator:
    def __init__(self):
        pass

    def string_to_ascii(self, s):
        return np.array([ord(c) for c in s]) 

    def ascii_to_string(self, ascii_list):
        return ''.join([chr(c) for c in ascii_list])

    def dynamic_time_warping(self, strings):
        ascii_strings = [self.string_to_ascii(s) for s in strings]
        reference_sequence = ascii_strings[0]
        common_indices = set(range(len(reference_sequence)))

        for i in range(1, len(ascii_strings)):
            sequence = ascii_strings[i]
            _, path = fastdtw(reference_sequence, sequence)
            matched_indices = {ref_idx for ref_idx, seq_idx in path if reference_sequence[ref_idx] == sequence[seq_idx]}
            common_indices &= matched_indices

        common_ascii = [reference_sequence[i] for i in sorted(common_indices)]
        common_string = self.ascii_to_string(common_ascii)
        return self.substitute_non_matching(strings[0], common_string)

    def substitute_non_matching(self, string, common_part):
        result = []
        common_idx = 0
        inside_placeholder = False

        for char in string:
            if common_idx < len(common_part) and char == common_part[common_idx]:
                result.append(char)
                common_idx += 1
                inside_placeholder = False
            else:
                if not inside_placeholder:
                    result.append("<*>")
                    inside_placeholder = True

        return ''.join(result)

    def combine_consecutive_star(self, common_part):
        final_result = ''.join(common_part)
        final_result = re.sub(r'(<\*>\s+)+<\*>', '<*>', final_result)  
        return final_result

    def postprocess(self, template):
        tokens = template.split(' ')
        processed_tokens = []

        for token in tokens:
            processed_token = ''
            i = 0
            star_exsit = False
            while i < len(token):
                if token[i:i+3] == "<*>":
                    star_exsit = True
                i += 1

            if star_exsit:
                i = 0
                while i < len(token):
                    if token[i:i+3] == "<*>":
                        left = i - 1
                        right = i + 3
                        while left >= 0 and token[left] not in ":=([{":
                            left -= 1
                        while right < len(token) and token[right] not in ")]}":
                            right += 1
                        processed_token = token[:left + 1] + '<*>' + token[right:]
                        i = right
                    else:
                        i += 1
            else:
                processed_token = token
            processed_tokens.append(processed_token)

        return ' '.join(processed_tokens)
