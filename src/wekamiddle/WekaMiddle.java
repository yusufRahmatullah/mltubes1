/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekamiddle;

import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author YusufR
 */
public class WekaMiddle {

    private static Instances data;
    private static String result;
    private static Classifier classifier;
    private static double percent = 66.0;
    private static Instances testSet;
    private static float elapsedTime;
    private static Evaluation evaluation;
    private static String evaluationText;
    private static String classifyResult;
    private static boolean isClassifyResultShow = false;
    
    public static void openFile(File file) {
        try {
            DataSource source = new DataSource(file.getPath());
            data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void saveFile(File file) {
        try {
            SerializationHelper.write(file.getPath(), data);
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void removeAttributes(int[] attributeIndices) {
        try {
            //debug
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(attributeIndices);
            remove.setInputFormat(data);
            data = weka.filters.Filter.useFilter(data, remove);
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void resampleAtribute() {
        // for supervised learning
        if(!data.instance(0).classIsMissing()) {
            weka.filters.supervised.instance.Resample resample = 
                new weka.filters.supervised.instance.Resample();
            try {
                resample.setInputFormat(data);
                data = weka.filters.supervised.instance.Resample.useFilter(data, resample);
            } catch (Exception ex) {
                Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        // for unsupervised learning
        else {
            weka.filters.unsupervised.instance.Resample resample = 
                new weka.filters.unsupervised.instance.Resample();
            try {
                resample.setInputFormat(data);
                data = weka.filters.unsupervised.instance.Resample.useFilter(data, resample);
            } catch (Exception ex) {
                Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
    
    public static void setClassifier(String classifierText) {
        if(classifierText.equalsIgnoreCase("NaiveBayes")) {
            classifier = new NaiveBayes();
        } else if (classifierText.equalsIgnoreCase("J48")) {
            classifier = new J48();
        } else if (classifierText.equalsIgnoreCase("IDTree")) {
            classifier = new Id3();
        } else if (classifierText.equalsIgnoreCase("myID3")) {
            classifier = new myID3();
        } else if (classifierText.equalsIgnoreCase("myJ48")) {
            classifier = new myJ48();
        }
    }
    
    public static void setEvaluation (String _evaluationText) {
        evaluationText = _evaluationText;
    }
    
    public static void setPercent(double newPercent) {
        if (newPercent >=0 && newPercent <= 100) {
            percent = newPercent;
        }
    }
    
    public static void setTestSet (File file) {
        try {
            DataSource source = new DataSource(file.getPath());
            testSet = source.getDataSet();
            testSet.setClassIndex(testSet.numAttributes() - 1);
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void evaluate() {
        try {
            float startTime = System.currentTimeMillis();
            classifier.buildClassifier(data);
            evaluation = new Evaluation (data);
            if (evaluationText.equalsIgnoreCase("10-fold cross validation")) {
                evaluation.crossValidateModel(classifier, data, 10, new Random(1));
            } else if (evaluationText.equalsIgnoreCase("Percentage split")) {
                int trainSize = (int) Math.round(data.numInstances() * percent / 100);
                int testSize = data.numInstances() - trainSize;
                Instances train = new Instances(data, 0, trainSize);
                Instances test = new Instances(data, trainSize, testSize);
                //debug
                System.out.println("train: "+train.numInstances());
                System.out.println("test: "+test.numInstances());
                
                classifier.buildClassifier(train);
                evaluation = new Evaluation(test);
                evaluation.evaluateModel(classifier, test);
            } else if (evaluationText.equalsIgnoreCase("Classify unlabeled data")) {
                classifier.buildClassifier(data);
                //evaluation = new Evaluation(testSet);
                //evaluation.evaluateModel(classifier, testSet);
                if (isClassifyResultShow) {
                    classifyResult = "";
                    for (int i=0; i<testSet.numInstances(); i++) {
                        double res = classifier.classifyInstance(testSet.instance(i));
                        String insText = testSet.instance(i).toString();
                        insText = insText.replaceAll("\\?", testSet.instance(i).classAttribute().value((int)res));
                        classifyResult += insText + "\n";
                    }
                }
            }else if (evaluationText.equalsIgnoreCase("Supplied test set")) {
                classifier.buildClassifier(data);
                evaluation = new Evaluation(testSet);
                evaluation.evaluateModel(classifier, testSet);
            } else {
                evaluation.evaluateModel(classifier, data);
            }
            elapsedTime = System.currentTimeMillis() - startTime;
            
            // set result text
            setResult();
            
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void saveModel(File file) {
        try {
            SerializationHelper.write(file.getPath(), classifier);
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static void loadModel (File file) {
        try {
            classifier = (Classifier)SerializationHelper.read(file.getPath());
        } catch (Exception ex) {
            Logger.getLogger(WekaMiddle.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static ArrayList<String> getAttributeNames () {
        ArrayList<String> attribs = new ArrayList<>();
        for (int i=0; i<data.numAttributes(); i++) {
            attribs.add(data.attribute(i).name());
        }
        return attribs;
    }
    
    public static String[] getInstances() {
        String[] instances = new String[data.numInstances()];
        for (int i=0; i<data.numInstances(); i++) {
            //debug
            instances[i] = data.instance(i).toString();
        }
        return instances;
    }
    
    public static void reset() {
        testSet = null;
        result = null;
        classifyResult = null;
        classifier = null;
        percent = 66;
    }

    private static void setResult() throws Exception {
        result = "";
        result += "=== Classifier model (full tarining set) ===\n\n";
        result += classifier.toString()+"\n";
        result += "=== Evaluation on training set===\n";
        result += "Time taken to build model: " + elapsedTime / 1000 + " seconds\n";
        result += "=== Summary ===\n";
        result += evaluation.toSummaryString() + "\n" + 
                evaluation.toClassDetailsString() + "\n" +
                evaluation.toMatrixString();
        if (isClassifyResultShow) {
            result += "\n\nClassify Result:\n" + classifyResult;
        }
    }
    
    public static String getResult() {
        return result;
    }
    
    public static void setClassifyResultView(boolean flag) {
        isClassifyResultShow = flag;
    }
}
