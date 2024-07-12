import Levenshtein
from codebleu import calc_codebleu

ref = '''
  public void run() {\n\n  long tripleCounter = 0;\n  long startProcessingTime;\n  long endProcessingTime;\n
'''
pred = '''
  public void run() {\n\n  long tripleCounter = 0;\n  long startProcessingTime;\n  long endProcessingTime;\n
'''

string = """
"""

print(calc_codebleu([ref], [pred], lang="java")['codebleu'])
print(Levenshtein.distance(ref, pred))

