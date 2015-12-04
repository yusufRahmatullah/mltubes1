package testmyj48;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Statistics;
import weka.core.Utils;
import weka.core.Capabilities.Capability;

public class myJ48ClassifierTree {
  /** for serialization */
  static final long serialVersionUID = -4813820170260388194L;
  /** The model selection method. */
  protected myJ48ModelSelect m_toSelectModel;
  /** The training instances. */
  protected Instances m_train;
  /** Local model at node. */
  protected myJ48Split m_localModel;
  /** References to sons. */
  protected myJ48ClassifierTree [] m_sons;
  /** True if node is leaf. */
  protected boolean m_isLeaf;
  /** True if node is empty. */
  protected boolean m_isEmpty;
  /** The confidence factor for pruning. */
  float m_CF = 0.25f;

  public myJ48ClassifierTree(myJ48ModelSelect toSelectLocModel, float cf) throws Exception {
	m_toSelectModel = toSelectLocModel;
    m_CF = cf;
  }

  public void buildClassifier(Instances data) throws Exception {
   // remove instances with missing class
   data = new Instances(data);
   data.deleteWithMissingClass();
   buildTree(data);
   collapse();
   prune();
   cleanup(new Instances(data, 0));
  }

  public final void collapse(){
    double errorsOfSubtree;
    double errorsOfTree;
    int i;
    if (!m_isLeaf){
      errorsOfSubtree = getTrainingErrors();
      errorsOfTree = m_localModel.distribution().numIncorrect();
      if (errorsOfSubtree >= errorsOfTree-1E-3){
		// Free adjacent trees
		m_sons = null;
		m_isLeaf = true;
		// Get NoSplit Model for tree.
		m_localModel = new NoSplit(m_localModel.distribution());
      }else
    	  for (i=0;i<m_sons.length;i++)
    		  m_sons[i].collapse();
    }
  }

  /**
   * Prunes a tree using C4.5's pruning procedure.
   *
   * @throws Exception if something goes wrong
   */
  public void prune() throws Exception {
    double errorsLargestBranch;
    double errorsLeaf;
    double errorsTree;
    int indexOfLargestBranch;
    myJ48ClassifierTree largestBranch;
    int i;
    if (!m_isLeaf){
      // Prune all subtrees.
      for (i=0;i<m_sons.length;i++) m_sons[i].prune();
      // Compute error for largest branch
      indexOfLargestBranch = m_localModel.distribution().maxBag();
      errorsLargestBranch = Double.MAX_VALUE;
      // Compute error if this Tree would be leaf
      errorsLeaf = getEstimatedErrorsForDistribution(m_localModel.distribution());
      // Compute error for the whole subtree
      errorsTree = getEstimatedErrors();
      // Decide if leaf is best choice.
      if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
    	  Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){
			// Free son Trees
			m_sons = null;
			m_isLeaf = true;
			// Get NoSplit Model for node.
			m_localModel = new NoSplit(m_localModel.distribution());
			return;
      }
      // Decide if largest branch is better choice
      // than whole subtree.
      if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
		largestBranch = m_sons[indexOfLargestBranch];
		m_sons = largestBranch.m_sons;
		m_localModel = largestBranch.m_localModel;
		m_isLeaf = largestBranch.m_isLeaf;
		newDistribution(m_train);
		prune();
      }
    }
  }

//  protected myJ48ClassifierTree getNewTree(Instances data) throws Exception {
//    myJ48ClassifierTree newTree = new myJ48ClassifierTree(m_toSelectModel, m_CF);
//    newTree.buildTree((Instances)data, false);
//    return newTree;
//  }
//
  private double getEstimatedErrors(){
    double errors = 0;
    int i;
    if (m_isLeaf)
      return getEstimatedErrorsForDistribution(m_localModel.distribution());
    else{
      for (i=0;i<m_sons.length;i++)
    	  errors = errors+m_sons[i].getEstimatedErrors();
      return errors;
    }
  }
//  
//  /**
//   * Computes estimated errors for one branch.
//   *
//   * @param data the data to work with
//   * @return the estimated errors
//   * @throws Exception if something goes wrong
//   */
//  private double getEstimatedErrorsForBranch(Instances data) 
//       throws Exception {
//
//    Instances [] localInstances;
//    double errors = 0;
//    int i;
//
//    if (m_isLeaf)
//      return getEstimatedErrorsForDistribution(new Distribution(data));
//    else{
//      Distribution savedDist = m_localModel.m_distribution;
//      m_localModel.resetDistribution(data);
//      localInstances = (Instances[])m_localModel.split(data);
//      m_localModel.m_distribution = savedDist;
//      for (i=0;i<m_sons.length;i++)
//	errors = errors+
//	  son(i).getEstimatedErrorsForBranch(localInstances[i]);
//      return errors;
//    }
//  }
//
  private double getEstimatedErrorsForDistribution(Distribution theDistribution){
    if (Utils.eq(theDistribution.total(),0))
      return 0;
    else
      return theDistribution.numIncorrect()
    		  + myJ48.addErrs(theDistribution.total(), theDistribution.numIncorrect(),m_CF);
  }

  private double getTrainingErrors(){
    double errors = 0;
    int i;
    if (m_isLeaf)
      return m_localModel.distribution().numIncorrect();
    else{
      for (i=0;i<m_sons.length;i++)
    	  errors = errors+m_sons[i].getTrainingErrors();
      return errors;
    }
  }

  private void newDistribution(Instances data) throws Exception {
    Instances [] localInstances;
    m_localModel.resetDistribution(data);
    m_train = data;
    if (!m_isLeaf){
      localInstances = (Instances [])m_localModel.split(data);
      for (int i = 0; i < m_sons.length; i++)
    	  m_sons[i].newDistribution(localInstances[i]);
    } else {
      // Check whether there are some instances at the leaf now!
      if (!Utils.eq(data.sumOfWeights(), 0)) {
    	  m_isEmpty = false;
      }
    }
  }
  public void buildTree(Instances data) throws Exception {
    Instances [] localInstances;
//    m_test = null;
    m_isLeaf = false;
    m_isEmpty = false;
    m_sons = null;
    m_localModel = m_toSelectModel.selectModel(data);
    if (m_localModel.numSubsets() > 1) {
      localInstances = m_localModel.split(data);
      data = null;
      m_sons = new myJ48ClassifierTree [m_localModel.numSubsets()];
      for (int i = 0; i < m_sons.length; i++) {
		m_sons[i] = getNewTree(localInstances[i]);
		localInstances[i] = null;
      }
    }else{
      m_isLeaf = true;
      if (Utils.eq(data.sumOfWeights(), 0))
	m_isEmpty = true;
      data = null;
    }
  }

  protected myJ48ClassifierTree getNewTree(Instances data) throws Exception {
    myJ48ClassifierTree newTree = new myJ48ClassifierTree(m_toSelectModel, m_CF);
    newTree.buildTree(data);
    return newTree;
  }

  public final void cleanup(Instances justHeaderInfo) {
    m_train = justHeaderInfo;
//    m_test = null;
    if (!m_isLeaf)
      for (int i = 0; i < m_sons.length; i++)
    	  m_sons[i].cleanup(justHeaderInfo);
  }

  public double classifyInstance(Instance instance) throws Exception {
    double maxProb = -1;
    double currentProb;
    int maxIndex = 0;
    int j;
    for (j = 0; j < instance.numClasses(); j++) {
      currentProb = getProbs(j, instance, 1);
      if (Utils.gr(currentProb,maxProb)) {
		maxIndex = j;
		maxProb = currentProb;
      }
    }
    return (double)maxIndex;
  }

  private double getProbs(int classIndex, Instance instance, double weight) throws Exception {
    double prob = 0;
    if (m_isLeaf) {
      return weight * m_localModel.classProb(classIndex, instance, -1);
    } else {
      int treeIndex = m_localModel.whichSubset(instance);
      if (treeIndex == -1) {
		double[] weights = m_localModel.weights(instance);
		for (int i = 0; i < m_sons.length; i++) {
		  if (!m_sons[i].m_isEmpty) {
		    prob += m_sons[i].getProbs(classIndex, instance, weights[i] * weight);
		  }
		}
		return prob;
      } else {
		if (m_sons[treeIndex].m_isEmpty) {
		  return weight * m_localModel.classProb(classIndex, instance, treeIndex);
		} else {
		  return m_sons[treeIndex].getProbs(classIndex, instance, weight);
		}
      }
    }
  }
}
