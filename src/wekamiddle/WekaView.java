/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekamiddle;

import java.awt.Component;
import java.io.File;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.AbstractButton;
import javax.swing.ButtonGroup;
import javax.swing.ButtonModel;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

/**
 *
 * @author YusufR
 */
public class WekaView extends javax.swing.JFrame {

    /**
     * Creates new form WekaView
     */
    public WekaView() {
        // set look and feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException ex) {
            Logger.getLogger(WekaView.class.getName()).log(Level.SEVERE, null, ex);
        }
        // init component
        initComponents();
        
        // set window's title
        this.setTitle("Weka Access by 13512040 and 13512068");
        
        // set init value
        WekaMiddle.setClassifier("NaiveBayes");
        classifierGroup.getElements().nextElement().setSelected(true);
        WekaMiddle.setEvaluation("Use training set");
        optionsGroup.getElements().nextElement().setSelected(true);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        classifierGroup = new javax.swing.ButtonGroup();
        optionsGroup = new javax.swing.ButtonGroup();
        attributeScrollPane = new javax.swing.JScrollPane();
        removeAttributeButton = new javax.swing.JButton();
        jScrollPane2 = new javax.swing.JScrollPane();
        instancesView = new javax.swing.JTextArea();
        fileLabel = new javax.swing.JLabel();
        jPanel1 = new javax.swing.JPanel();
        bayesRadio = new javax.swing.JRadioButton();
        j48Radio = new javax.swing.JRadioButton();
        id3Radio = new javax.swing.JRadioButton();
        myId3Radio = new javax.swing.JRadioButton();
        myC45Radio = new javax.swing.JRadioButton();
        jPanel2 = new javax.swing.JPanel();
        trainRadio = new javax.swing.JRadioButton();
        testRadio = new javax.swing.JRadioButton();
        crossRadio = new javax.swing.JRadioButton();
        percentageRadio = new javax.swing.JRadioButton();
        percentageValue = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        testOpenFile = new javax.swing.JButton();
        classifyUnlabeledRadio = new javax.swing.JRadioButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        resultText = new javax.swing.JTextArea();
        jLabel2 = new javax.swing.JLabel();
        saveModel = new javax.swing.JButton();
        openModel = new javax.swing.JButton();
        modelLabel = new javax.swing.JLabel();
        classifyButton = new javax.swing.JButton();
        testSetLabel = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenu1 = new javax.swing.JMenu();
        openFile = new javax.swing.JMenuItem();
        saveFile = new javax.swing.JMenuItem();
        jSeparator1 = new javax.swing.JPopupMenu.Separator();
        exitMenu = new javax.swing.JMenuItem();
        jMenu2 = new javax.swing.JMenu();
        resample = new javax.swing.JMenuItem();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        removeAttributeButton.setText("Remove Atrributes");
        removeAttributeButton.setEnabled(false);
        removeAttributeButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                removeAttributeButtonActionPerformed(evt);
            }
        });

        instancesView.setColumns(20);
        instancesView.setRows(5);
        jScrollPane2.setViewportView(instancesView);

        fileLabel.setText("File: ");

        jPanel1.setBorder(javax.swing.BorderFactory.createTitledBorder("Classifier"));
        jPanel1.setToolTipText("");

        classifierGroup.add(bayesRadio);
        bayesRadio.setText("NaiveBayes");

        classifierGroup.add(j48Radio);
        j48Radio.setText("J48");

        classifierGroup.add(id3Radio);
        id3Radio.setText("IDTree");

        classifierGroup.add(myId3Radio);
        myId3Radio.setText("myID3");

        classifierGroup.add(myC45Radio);
        myC45Radio.setText("myC45");

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(bayesRadio)
                    .addComponent(j48Radio)
                    .addComponent(id3Radio)
                    .addComponent(myId3Radio)
                    .addComponent(myC45Radio))
                .addGap(0, 0, Short.MAX_VALUE))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(bayesRadio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(j48Radio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(id3Radio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(myId3Radio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(myC45Radio)
                .addContainerGap())
        );

        jPanel2.setBorder(javax.swing.BorderFactory.createTitledBorder("Test Options"));

        optionsGroup.add(trainRadio);
        trainRadio.setText("Use training set");
        trainRadio.setToolTipText("");

        optionsGroup.add(testRadio);
        testRadio.setText("Supplied test set");
        testRadio.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                testRadioStateChanged(evt);
            }
        });
        testRadio.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                testRadioActionPerformed(evt);
            }
        });

        optionsGroup.add(crossRadio);
        crossRadio.setText("10-fold cross validation");

        optionsGroup.add(percentageRadio);
        percentageRadio.setText("Percentage split");
        percentageRadio.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                percentageRadioStateChanged(evt);
            }
        });

        percentageValue.setText("66");
        percentageValue.setEnabled(false);

        jLabel1.setText("%");

        testOpenFile.setText("set");
        testOpenFile.setEnabled(false);
        testOpenFile.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                testOpenFileActionPerformed(evt);
            }
        });

        optionsGroup.add(classifyUnlabeledRadio);
        classifyUnlabeledRadio.setText("Classify unlabeled data");
        classifyUnlabeledRadio.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                classifyUnlabeledRadioStateChanged(evt);
            }
        });

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addComponent(percentageRadio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(percentageValue, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel1)
                .addGap(0, 0, Short.MAX_VALUE))
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(trainRadio)
                    .addComponent(crossRadio))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(testRadio)
                    .addComponent(classifyUnlabeledRadio))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(testOpenFile, javax.swing.GroupLayout.DEFAULT_SIZE, 70, Short.MAX_VALUE)
                .addGap(6, 6, 6))
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(trainRadio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel2Layout.createSequentialGroup()
                        .addComponent(testRadio)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(classifyUnlabeledRadio)
                        .addGap(4, 4, 4))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel2Layout.createSequentialGroup()
                        .addComponent(testOpenFile)
                        .addGap(15, 15, 15)))
                .addComponent(crossRadio)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(percentageRadio)
                    .addComponent(percentageValue, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1)))
        );

        resultText.setColumns(20);
        resultText.setRows(5);
        jScrollPane1.setViewportView(resultText);

        jLabel2.setText("Result:");

        saveModel.setText("Save model");
        saveModel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveModelActionPerformed(evt);
            }
        });

        openModel.setText("Open Model");
        openModel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                openModelActionPerformed(evt);
            }
        });

        modelLabel.setText("Model: ");

        classifyButton.setText("Classify");
        classifyButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                classifyButtonActionPerformed(evt);
            }
        });

        testSetLabel.setText("Test set:");

        jMenu1.setText("File");

        openFile.setText("Open File");
        openFile.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                openFileActionPerformed(evt);
            }
        });
        jMenu1.add(openFile);

        saveFile.setText("Save File");
        saveFile.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveFileActionPerformed(evt);
            }
        });
        jMenu1.add(saveFile);
        jMenu1.add(jSeparator1);

        exitMenu.setMnemonic('x');
        exitMenu.setText("Exit");
        exitMenu.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                exitMenuActionPerformed(evt);
            }
        });
        jMenu1.add(exitMenu);

        jMenuBar1.add(jMenu1);

        jMenu2.setText("Data");

        resample.setText("Resample Data");
        resample.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                resampleActionPerformed(evt);
            }
        });
        jMenu2.add(resample);

        jMenuBar1.add(jMenu2);

        setJMenuBar(jMenuBar1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(attributeScrollPane)
                    .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 245, Short.MAX_VALUE)
                    .addComponent(fileLabel)
                    .addComponent(removeAttributeButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 279, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(classifyButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(testSetLabel)
                                .addGap(0, 0, Short.MAX_VALUE))))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel2)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(saveModel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(openModel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(modelLabel)))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(33, 33, 33)
                        .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(testSetLabel)
                        .addGap(2, 2, 2)
                        .addComponent(classifyButton))
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(fileLabel)
                            .addComponent(jLabel2))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, 175, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(attributeScrollPane))
                            .addComponent(jScrollPane1))))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(removeAttributeButton)
                    .addComponent(saveModel)
                    .addComponent(openModel)
                    .addComponent(modelLabel))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void exitMenuActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_exitMenuActionPerformed
        this.setVisible(false);
        this.dispose();
    }//GEN-LAST:event_exitMenuActionPerformed

    private void openFileActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_openFileActionPerformed
        // load data
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            WekaMiddle.openFile(selectedFile);
            fileLabel.setText("File: "+selectedFile.getName());
            
            // reset test set, percentage, and classifier
            WekaMiddle.reset();
            resultText.setText(null);
            testSetLabel.setText("Test set:");
            modelLabel.setText("Model:");

            // show all attributes
            ArrayList<String> attribs = WekaMiddle.getAttributeNames();        
            CheckBoxList cbList = new CheckBoxList();
            JCheckBox[] jcbList = new JCheckBox[attribs.size()];
            for (int i=0; i<attribs.size(); i++) {
                jcbList[i] = new JCheckBox();
                jcbList[i].setText(attribs.get(i));
            }
            cbList.setListData(jcbList);
            cbList.setSize(attributeScrollPane.getSize());
            attributeScrollPane.setViewportView(cbList);

            // enable remove attributes button
            removeAttributeButton.setEnabled(true);

            // show all instances
            String[] instances = WekaMiddle.getInstances();
            String instancesText = "";
            for (String instance : instances) {
                instancesText += instance + "\n";
            }
            instancesView.setText(instancesText);
        }
    }//GEN-LAST:event_openFileActionPerformed

    private void removeAttributeButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_removeAttributeButtonActionPerformed
        CheckBoxList cbList = (CheckBoxList) attributeScrollPane.getViewport().getComponent(0);
        WekaMiddle.removeAttributes(cbList.getCheckBoxSelectedIndices());
        // re-list attributes
        // show all attributes
        ArrayList<String> attribs = WekaMiddle.getAttributeNames();        
        CheckBoxList newcbList = new CheckBoxList();
        JCheckBox[] jcbList = new JCheckBox[attribs.size()];
        for (int i=0; i<attribs.size(); i++) {
            jcbList[i] = new JCheckBox();
            jcbList[i].setText(attribs.get(i));
        }
        newcbList.setListData(jcbList);
        newcbList.setSize(attributeScrollPane.getSize());
        attributeScrollPane.setViewportView(newcbList);
        attributeScrollPane.repaint();
        
        // show all instances
        String[] instances = WekaMiddle.getInstances();
        String instancesText = "";
        for (String instance : instances) {
            instancesText += instance + "\n";
        }
        instancesView.setText(instancesText);
    }//GEN-LAST:event_removeAttributeButtonActionPerformed

    private void resampleActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_resampleActionPerformed
        WekaMiddle.resampleAtribute();
        
        // show all instances
        String[] instances = WekaMiddle.getInstances();
        String instancesText = "";
        for (String instance : instances) {
            instancesText += instance + "\n";
        }
        instancesView.setText(instancesText);
    }//GEN-LAST:event_resampleActionPerformed

    private void testRadioStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_testRadioStateChanged
        if (testRadio.isSelected()) {
            testOpenFile.setEnabled(true);
        } else {
            testOpenFile.setEnabled(false);
        }
    }//GEN-LAST:event_testRadioStateChanged

    private void percentageRadioStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_percentageRadioStateChanged
        if (percentageRadio.isSelected()) {
            percentageValue.setEnabled(true);
        } else {
            percentageValue.setEnabled(false);
        }
    }//GEN-LAST:event_percentageRadioStateChanged

    private void classifyButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_classifyButtonActionPerformed
        String classifierText = "";
        String optionsText = "";
        for(Enumeration<AbstractButton> buttons = classifierGroup.getElements(); buttons.hasMoreElements();){
            AbstractButton button = buttons.nextElement();
            if (button.isSelected()) {
                classifierText = button.getText();
                break;
            }
        }
        for (Enumeration<AbstractButton> buttons = optionsGroup.getElements(); buttons.hasMoreElements();) {
            AbstractButton button = buttons.nextElement();
            if (button.isSelected()) {
                optionsText = button.getText();
                break;
            }
        }
        WekaMiddle.setPercent(Double.valueOf(percentageValue.getText()));
        WekaMiddle.setClassifier(classifierText);
        WekaMiddle.setEvaluation(optionsText);
        WekaMiddle.evaluate();
        
        // set result text
        resultText.setText(WekaMiddle.getResult());
    }//GEN-LAST:event_classifyButtonActionPerformed

    private void testOpenFileActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_testOpenFileActionPerformed
        // load test set
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            WekaMiddle.setTestSet(selectedFile);
            testSetLabel.setText("Test set: "+selectedFile.getName());
        }
    }//GEN-LAST:event_testOpenFileActionPerformed

    private void openModelActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_openModelActionPerformed
        // load model
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            WekaMiddle.loadModel(selectedFile);
            modelLabel.setText("Model: "+selectedFile.getName());
        }
    }//GEN-LAST:event_openModelActionPerformed

    private void saveModelActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveModelActionPerformed
        // save model
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        int result = fileChooser.showSaveDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            WekaMiddle.saveFile(selectedFile);
        }
    }//GEN-LAST:event_saveModelActionPerformed

    private void saveFileActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveFileActionPerformed
        // save model
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
        int result = fileChooser.showSaveDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            WekaMiddle.loadModel(selectedFile);
            modelLabel.setText("Model: "+selectedFile.getName());
        }
    }//GEN-LAST:event_saveFileActionPerformed

    private void testRadioActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_testRadioActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_testRadioActionPerformed

    private void classifyUnlabeledRadioStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_classifyUnlabeledRadioStateChanged
        if (classifyUnlabeledRadio.isSelected()) {
            testOpenFile.setEnabled(true);
        } else {
            testOpenFile.setEnabled(false);
        }
    }//GEN-LAST:event_classifyUnlabeledRadioStateChanged

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(WekaView.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(WekaView.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(WekaView.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(WekaView.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new WekaView().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JScrollPane attributeScrollPane;
    private javax.swing.JRadioButton bayesRadio;
    private javax.swing.ButtonGroup classifierGroup;
    private javax.swing.JButton classifyButton;
    private javax.swing.JRadioButton classifyUnlabeledRadio;
    private javax.swing.JRadioButton crossRadio;
    private javax.swing.JMenuItem exitMenu;
    private javax.swing.JLabel fileLabel;
    private javax.swing.JRadioButton id3Radio;
    private javax.swing.JTextArea instancesView;
    private javax.swing.JRadioButton j48Radio;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JMenu jMenu1;
    private javax.swing.JMenu jMenu2;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JPopupMenu.Separator jSeparator1;
    private javax.swing.JLabel modelLabel;
    private javax.swing.JRadioButton myC45Radio;
    private javax.swing.JRadioButton myId3Radio;
    private javax.swing.JMenuItem openFile;
    private javax.swing.JButton openModel;
    private javax.swing.ButtonGroup optionsGroup;
    private javax.swing.JRadioButton percentageRadio;
    private javax.swing.JTextField percentageValue;
    private javax.swing.JButton removeAttributeButton;
    private javax.swing.JMenuItem resample;
    private javax.swing.JTextArea resultText;
    private javax.swing.JMenuItem saveFile;
    private javax.swing.JButton saveModel;
    private javax.swing.JButton testOpenFile;
    private javax.swing.JRadioButton testRadio;
    private javax.swing.JLabel testSetLabel;
    private javax.swing.JRadioButton trainRadio;
    // End of variables declaration//GEN-END:variables
}