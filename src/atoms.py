'''
Molecular Parser : Counting atoms from a molecular formula
    For example:
        - The input 'H2O' must return {'H': 2, 'O': 1}
        - The input 'Mg(OH)2' must return {'Mg': 1, 'O': 2, 'H': 2}
        - The input 'K4[ON(SO3)2]2' must return {'K': 4, 'O': 14, 'N': 2, 'S': 4}
Algorithm explanation graph : https://tinyurl.com/w8fg2cy
Author : 
    yacine-benbaccar [yacine.benbaccar at telecom-paris.fr]
Useful Resources : 
    - https://regexr.com/
    - https://www.w3schools.com/python/python_regex.asp
Licence:
    MIT License
    Copyright (c) [2019] [yacine-benbaccar]
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

import re

class MolecularDecomposer(object):
    def __init__(self, inp, verbose=False):
        self.inp = inp
        self.verbose = verbose
        self.result = {}
        # Mollecular Pattern
        self.molPattern = r"[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+"

    def __addToDict__(self,aux,multiplier=1):
        if aux not in self.result.keys():
            self.result[aux] = multiplier
        else:
            self.result[aux] += multiplier
    
    def __findLastDigit__(self, aux):
        lastDigit = ""
        for i in range(-1,-len(aux),-1):
            if aux[i].isdigit():
                lastDigit+=aux[i]
            else:
                break
        if len(lastDigit) ==0:
            return 1 , 0
        else:
            return int(''.join(reversed(lastDigit))), len(lastDigit)

    def __simpleDecompose__(self):
        # Replace brackets and braces with parenthesis (to avoid complex re)
        aux = re.sub(r'[\[\{]',r'(',self.inp)
        aux = re.sub(r'[\]\}]',r')', aux)

        if self.verbose:
            print('The new formula is : ', aux)

        return re.findall(self.molPattern, aux)


    def ShatterFormula(self):
        if self.verbose:
            print(".... Shattering the {} formula ....".format(self.inp))
        x = self.__simpleDecompose__()
        if self.verbose:
            print("First level decomposition : ", x)
        k = 0
        sizes = []
        # Shatter the formula
        while(len(x)>0):
            k+=1
            sizes.append(len(x))
            for i,e in enumerate(x):
                if '(' and ')' in e:
                    multiplier, n = self.__findLastDigit__(e)
                    aux = e[:-n]
                    aux = aux[1:-1]
                    x.pop(i)
                    tmp = re.findall(self.molPattern, aux)*multiplier
                    if self.verbose:
                        print("----Decomposing {} into :".format(e), tmp)
                    x.extend(tmp)
                else:
                    # Count atoms
                    multiplier, n = self.__findLastDigit__(e)
                    if n == 0:
                        self.__addToDict__(e, multiplier=multiplier)
                    else:
                        self.__addToDict__(e[:-n], multiplier=multiplier)
                    x.pop(i)
            
        if self.verbose:
            print("#iterations : ", k)
            # print("size of x at each iteration : ", sizes)
            # Count the atoms
            print(".... Counting atoms ....\n")
        return self.result

if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        print("------ Testing the provided Formulas (3 Test Cases) ------")
        tests = ['H2O','Mg(OH)2','K4[ON(SO3)2]2']
        print([(t,MolecularDecomposer(t, verbose=True).ShatterFormula()) for t in tests])
        print("Done!\n")

    molecule = input("Enter molecular formula (Type 'quit' to exit program): ")
    while(molecule not in ['quit', '!q']):
        print((molecule, MolecularDecomposer(molecule, verbose=True).ShatterFormula()))
        molecule = input("Enter molecular formula (Type 'quit' to exit program): ")
