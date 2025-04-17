
import React from 'react';
import { cn } from '@/lib/utils';

interface ProcessingStepProps {
  number: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

const ProcessingStep: React.FC<ProcessingStepProps> = ({
  number,
  title,
  description,
  icon,
  className,
  onClick,
}) => {
  return (
    <div 
      className={cn("step-card", className)}
      onClick={onClick}
    >
      <div className="step-number bg-primary/10 text-primary font-medium rounded-full w-6 h-6 flex items-center justify-center text-sm mb-3">{number}</div>
      <div className="processing-step-icon text-primary mb-2">
        {icon}
      </div>
      <h3 className="text-lg font-semibold mb-1">{title}</h3>
      <p className="text-muted-foreground text-sm">{description}</p>
    </div>
  );
};

export default ProcessingStep;
