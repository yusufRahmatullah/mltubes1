package testmyj48;

import weka.core.Instance;
import weka.core.Instances;

public final class NoSplit extends myJ48Split{

  public NoSplit(Distribution distribution){
    super(-1,-1,-1);
    m_distribution = new Distribution(distribution);
    m_numSubsets = 1;
  }

  public final void buildClassifier(Instances instances) throws Exception {
    m_distribution = new Distribution(instances);
    m_numSubsets = 1;
  }

  public int whichSubset(Instance instance){
    return 0;
  }

  public double [] weights(Instance instance){
    return null;
  }
}
