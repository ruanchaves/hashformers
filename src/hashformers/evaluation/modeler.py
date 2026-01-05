#!/usr/bin/python

#
# Modeler.py
# 
# Copyright (c) 2016, Bogazici University. See the file
# COPYRIGHT.txt in the software or at http://tabilab.cmpe.boun.edu.tr/projects/hashtag_segmentation/copyright.html
# 
# This file is part of BOUN Hashtag Segmentor (see http://tabilab.cmpe.boun.edu.tr/projects/hashtag_segmentation),
# and is free software, licensed under the GNU Library General Public License, Version 2, June 1991
# (in the distribution as file licence.html, and also available at
# http://tabilab.cmpe.boun.edu.tr/projects/hashtag_segmentation/licence.html).
# 
# Arda Celebi, TABI Labs, CMPE, Bogazici University, 
#

class Modeler:
    """Evaluation metrics modeler with per-instance state.
    
    Note: Class refactored from class-level mutable state to instance attributes
    for thread-safety (HASH-008).
    """
    
    def __init__(self):
        """Initialize instance with its own state (HASH-008: Thread-safe instance attributes)."""
        self.hashtagSegmentor = None
        self.t = 0
        self.totals = 0
        self.totalh = 0
        self.p = 0
        self.r = 0
        self.n = 0
        self.modelerParams = {}
    
    def loadParameters(self, args):
        leftoverArgs = []
        for arg in args:
            if self.loadParameter(arg) == False:
                leftoverArgs.append(arg)    
        return leftoverArgs
    
    def loadParameter(self, param):
        pass
    
    def getRunCode(self):
        return ""
    
    def train(self, featureFile):
        pass
    
    def segmentHashtag(self, hashtag):
        pass
    
    def segmentFile(self, fileToSegment, featureFileName, params):
        pass
    
    def calculateScore(self, testFile, params):
        pass
    
    def loadModelerParams(self, params):
        pass
    
    def test(self, testFile, featureFileName, params):
        (acc, precision, recall, fscore) = self.calculateScore(testFile, featureFileName, params)
    
        print("MAXENT ACC %f PRE %f REC %f F1 %f\n" % (acc, precision, recall, fscore))
        
    def isFeatureOn(self, feature):
        return False
    
    def reset(self):
        self.t = 0
        self.totals = 0
        self.totalh = 0
        self.p = 0
        self.r = 0
        self.n = 0
    
    def countEntry(self, segmented, trueSegmentation):
        sw = segmented.split(' ')
        hw = trueSegmentation.split(' ')
        for s in sw:
            for h in hw:
                if s == h:
                    self.p = self.p + 1
                    break
        for h in hw:
            for s in sw:
                if s == h:
                    self.r = self.r + 1
                    break
                
        self.totals = self.totals + len(sw)
        self.totalh = self.totalh + len(hw)
        self.n += 1
        
        if segmented == trueSegmentation:
            self.t += 1
        
    def calculatePrecision(self):
        if self.totals > 0:
            # Modernized Python casting style (HASH-013)
            return float(self.p * 100) / float(self.totals)
        return 0
    
    def calculateRecall(self):
        if self.totalh > 0:
            # Modernized Python casting style (HASH-013)
            return float(self.r * 100) / float(self.totalh)
        return 0
    
    def calculateFScore(self):
        precision = self.calculatePrecision()
        recall = self.calculateRecall()
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0
    
    def calculateAccuracy(self):
        if self.n > 0:
            # Modernized Python casting style (HASH-013)
            return float(100 * self.t) / float(self.n)
        return 0
