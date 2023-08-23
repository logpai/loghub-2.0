import os
from logparser.Logram.DictionarySetUp import dictionaryBuilder
from logparser.Logram.MatchToken import tokenMatch


class LogParser:
    def __init__(self, indir, outdir, log_format , doubleThreshold, triThreshold, rex=[] ):
        self.indir = indir
        self.outdir = outdir
        self.doubleThreshold = doubleThreshold
        self.triThreshold = triThreshold
        self.log_format = log_format
        self.rex = rex

    def parse(self, log_file_basename):
        log_file = os.path.join(self.indir, log_file_basename)
        print('Parsing file: ' + log_file)
        doubleDictionaryList, triDictionaryList, allTokenList, allMessageList = dictionaryBuilder(
            self.log_format, log_file, self.rex
        )
        tokenMatch(
            allTokenList, doubleDictionaryList, triDictionaryList, self.doubleThreshold, self.triThreshold,
            self.outdir, log_file_basename, allMessageList
        )
