import re
import pandas as pd
import sys 

def parse_log (caminho):
  padrao =  r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<datetime>[^\]]+)]'
  
  print(f"Loading file {caminho}")
  with open(caminho) as file:
    for line in file:
        result = re.match(padrao, line)
        sys.exit(0)
  return result

if __name__ == "__main__":
   path = "logpai\Linux\Linux_2k.log"
   parse_log(path)