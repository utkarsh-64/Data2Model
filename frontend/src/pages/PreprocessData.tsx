
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import SectionHeader from '@/components/SectionHeader';
import { Layers } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Import refactored components
import WorkflowSteps from '@/components/preprocessing/WorkflowSteps';
import UploadStep from '@/components/preprocessing/UploadStep';
import DataSummary from '@/components/preprocessing/DataSummary';
import ExploratoryDataAnalysis from '@/components/preprocessing/ExploratoryDataAnalysis';
import DataCleaning from '@/components/preprocessing/DataCleaning';
import FeatureEngineering from '@/components/preprocessing/FeatureEngineering';
import ExportData from '@/components/preprocessing/ExportData';

type StepType = 'upload' | 'eda' | 'cleaning' | 'feature' | 'export';

const PreprocessData = () => {
  const { toast } = useToast();
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [activeStep, setActiveStep] = useState<StepType>('upload');
  
  const handleLoadData = () => {
    setIsDataLoaded(true);
    setActiveStep('eda');
    toast({
      title: "Data Loaded Successfully",
      description: "Your dataset has been loaded with 1,245 records and 7 columns.",
      className: "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-900/50",
    });
  };

  const handleReset = () => {
    setIsDataLoaded(false);
    setActiveStep('upload');
  };

  const handleStepChange = (step: StepType) => {
    if (isDataLoaded) {
      setActiveStep(step);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto animate-fade-in">
      <SectionHeader 
        title="Preprocess Data" 
        description="Clean, transform, and prepare your data for model training"
        icon={<Layers className="h-5 w-5" />}
      />
      
      {/* Workflow Progress Steps */}
      <WorkflowSteps 
        activeStep={activeStep} 
        isDataLoaded={isDataLoaded} 
        onStepClick={handleStepChange} 
      />
      
      {!isDataLoaded ? (
        <UploadStep onDataLoaded={handleLoadData} />
      ) : (
        <div className="space-y-8">
          <DataSummary onReset={handleReset} />
          
          {activeStep === 'eda' && (
            <ExploratoryDataAnalysis onNext={() => setActiveStep('cleaning')} />
          )}
          
          {activeStep === 'cleaning' && (
            <DataCleaning onNext={() => setActiveStep('feature')} />
          )}
          
          {activeStep === 'feature' && (
            <FeatureEngineering onNext={() => setActiveStep('export')} />
          )}
          
          {activeStep === 'export' && (
            <ExportData onPrevious={() => setActiveStep('feature')} />
          )}
        </div>
      )}
    </div>
  );
};

export default PreprocessData;
