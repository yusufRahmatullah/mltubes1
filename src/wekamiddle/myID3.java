/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekamiddle;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;

/**
 *
 * @author YusufR
 */
public class myID3 extends Classifier {
    
    private static final double log2 = Math.log(2);   // simplify
    private Instances data;
    private myID3[] nextNodes;  // successor nodes
    private Attribute attributeValue;   // value if this node is not leaf
    private double classValue;  // value if this node is leaf
    private Attribute classAttribute;   // attribute of leaf
    
    //TODO: penanganan binary class dan multi class
    //TODO: penanganan atribut diskrit dan kontinu
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        
        // check if data instances can be handle by classifier
        getCapabilities().testWithFail(instances);
        
        // remove instances with missing class
        data = new Instances(instances);
        data.deleteWithMissingClass();
        
        // if data is numeric, discretize data using filter: discertize
        if (data.classAttribute().isNumeric() || data.attribute(0).isNumeric()) {
            weka.filters.supervised.attribute.Discretize discretizeFilter = 
                    new weka.filters.supervised.attribute.Discretize();
            discretizeFilter.setInputFormat(data);
            data = weka.filters.supervised.attribute.Discretize.useFilter(data, discretizeFilter);
        }
        
        makeTree(data);
    }
    
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("myID3: no missing values, please.");
        }
        // if node is leaf return it's class value
        // else call successor node
        if (attributeValue == null) {
            return classValue;
        } else {
            return nextNodes[(int) instance.value(attributeValue)].classifyInstance(instance);
        }
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();
        
        // enable nominal attributes
        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        // enable numeric attributes
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        // enable nominal class
        capabilities.enable(Capability.NOMINAL_CLASS);
        // enable numeric class
        capabilities.enable(Capability.NUMERIC_CLASS);
        // enable missing class value
        capabilities.enable(Capability.MISSING_CLASS_VALUES);
        // set minimum instances number
        capabilities.setMinimumNumberInstances(0);
        
        return capabilities;
    }

    private void makeTree(Instances data) {
        
        // check if attributes empty, return single-node tree root, 
        // with label = most common value of TargetAttributes in Example
        // base of the recursive
        if (data.numInstances() == 0) {
            attributeValue = null;
            classValue = Instance.missingValue();
            return;
        }
        
        // compute attribute that has biggest information gain
        double[] ig = new double[data.numAttributes()];
        for (Enumeration attribEnum = data.enumerateAttributes(); attribEnum.hasMoreElements();) {
            Attribute attrib = (Attribute) attribEnum.nextElement();
            ig[attrib.index()] = infoGain(data, attrib);
        }
        int maxIdx = 0; // index of attribute with max information gain
        for (int i=0; i<data.numAttributes(); i++) {
            if (ig[i] > ig[maxIdx]) {
                maxIdx = i;
            }
        }
        // debug
        for (int i=0; i<ig.length; i++) {
            System.out.println(data.attribute(i).name()+": ig["+i+"]: "+ig[i]);
        }
        // set highest information attribute to attribute value
        attributeValue = data.attribute(maxIdx);
        
        // make leaf or make new tree
        // make leaf if information gain = 0
        // else make new tree
        if (ig[maxIdx] == 0) {
            attributeValue = null;  // not a node
            double[] classValueCount = new double[data.numClasses()];
            for (Enumeration instanceEnum = data.enumerateInstances(); instanceEnum.hasMoreElements();) {
                Instance instance = (Instance) instanceEnum.nextElement();
                classValueCount[(int)instance.classValue()]++;
            }
            // get index of highest class value to fill leaf
            int cvMaxIdx = 0;
            for (int i=0; i<data.numClasses(); i++) {
                if (classValueCount[cvMaxIdx] < classValueCount[i]) {
                    cvMaxIdx = i;
                }
            }
            classValue = cvMaxIdx;  // value of class
            classAttribute = data.classAttribute();
        } else {
            Instances[] splittedData = splitDataByAttrib(data, attributeValue);
            nextNodes = new myID3[attributeValue.numValues()];
            for (int i=0; i<nextNodes.length; i++) {
                nextNodes[i] = new myID3();
                nextNodes[i].makeTree(splittedData[i]);
            }
        }
    }

    private double infoGain(Instances data, Attribute attrib) {
        double ig = 0;
        double sigma = 0;
        // ig = entropy(S) - sigma(Sv/S * Entropy(Sv))
        Instances[] splittedDatas = splitDataByAttrib(data, attrib);
        for (Instances splittedData : splittedDatas) {
            if(splittedData.numInstances() > 0) {
                sigma += ((double) splittedData.numInstances() / (double) data.numInstances()) *
                        entropy(splittedData);
            }
        }
        ig = entropy(data) - sigma;
        //debug
        System.out.println("entropy: "+entropy(data)+" & sigma: "+sigma);
        return ig;
    }
   
    private double entropy(Instances data) {
        double entropy = 0;
        // entropy = -1 * p1 * log2(p1) + p2 * log2(p2) + ... + pn * log2(pn)
        double [] classProbs = new double[data.numClasses()];
        for (Enumeration instanceEnum = data.enumerateInstances(); instanceEnum.hasMoreElements();) {
            Instance instance = (Instance) instanceEnum.nextElement();
            classProbs[(int) instance.classValue()]++;
        }
        // calculate entropy from all class
        for (double classProb : classProbs) {
            if (classProb >0 ) {    // avoid log(0)
                classProb /= data.numInstances();
                entropy -= classProb * log2(classProb);
            }
        }
        
        return entropy;
    }

    private double log2(double classProb) {
        return Math.log(classProb) / log2;
    }

    private Instances[] splitDataByAttrib(Instances data, Attribute attrib) {
        // split data by it's attribut
        Instances[] splittedDatas = new Instances[attrib.numValues()];
        for (int i=0; i<attrib.numValues(); i++) {
            splittedDatas[i] = new Instances(data, data.numInstances());
        }
        for (Enumeration instanceEnum = data.enumerateInstances(); instanceEnum.hasMoreElements();) {
            Instance instance = (Instance) instanceEnum.nextElement();
            splittedDatas[(int)instance.value(attrib)].add(instance);
        }
        for (Instances splittedData : splittedDatas) {
            splittedData.compactify();
        }
        return splittedDatas;
    }
    
    @Override
    public String toString() {
        return "myID3\n\n" + toString(0);
    }
    
    public String toString(int level) {
        StringBuilder text = new StringBuilder();
        
        if (attributeValue == null) {
            if (Instance.isMissingValue(classValue)) {
                text.append(": null");
            } else {
                text.append(": "+classAttribute.value((int)classValue));
            }
        } else {
            for (int i=0; i<attributeValue.numValues(); i++) {
                text.append("\n");
                for (int j=0; j<level; j++) {
                    text.append("| ");
                }
                text.append(attributeValue.name() + " = " + attributeValue.value(i));
                text.append(nextNodes[i].toString(level+1));
            }
        }
        return text.toString();
    }
}
