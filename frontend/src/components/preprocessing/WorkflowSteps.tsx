
import React from 'react';
import { Upload, Search, Wrench, Columns, FileText } from 'lucide-react';
import ProcessingStep from '@/components/ProcessingStep';

type StepType = 'upload' | 'eda' | 'cleaning' | 'feature' | 'export';

interface WorkflowStepsProps {
  activeStep: StepType;
  isDataLoaded: boolean;
  onStepClick: (step: StepType) => void;
}

const WorkflowSteps: React.FC<WorkflowStepsProps> = ({ 
  activeStep, 
  isDataLoaded,
  onStepClick
}) => {
  const workflowSteps = [
    {
      id: 'upload' as StepType,
      title: 'Data Upload',
      description: 'Upload or select your dataset to begin processing',
      icon: <Upload className="h-5 w-5" />,
      number: 1
    },
    {
      id: 'eda' as StepType,
      title: 'Exploratory Data Analysis',
      description: 'Understand your data structure and patterns',
      icon: <Search className="h-5 w-5" />,
      number: 2
    },
    {
      id: 'cleaning' as StepType,
      title: 'Data Cleaning',
      description: 'Handle missing values and outliers',
      icon: <Wrench className="h-5 w-5" />,
      number: 3
    },
    {
      id: 'feature' as StepType,
      title: 'Feature Engineering',
      description: 'Transform and create new features for your model',
      icon: <Columns className="h-5 w-5" />,
      number: 4
    },
    {
      id: 'export' as StepType,
      title: 'Export Data',
      description: 'Export your processed data as CSV',
      icon: <FileText className="h-5 w-5" />,
      number: 5
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 gap-3 mb-6">
      {workflowSteps.map((step) => (
        <ProcessingStep
          key={step.id}
          number={step.number}
          title={step.title}
          description={step.description}
          icon={step.icon}
          className={`cursor-pointer transition-all p-4 border rounded-lg ${
            activeStep === step.id 
              ? 'border-primary bg-primary/5 ring-1 ring-primary' 
              : 'border-border hover:border-primary/40 hover:bg-muted/40'
          }`}
          onClick={() => isDataLoaded ? onStepClick(step.id) : null}
        />
      ))}
    </div>
  );
};

export default WorkflowSteps;
