
import React from 'react';
import { cn } from '@/lib/utils';

interface SectionHeaderProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
  className?: string;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  description,
  icon,
  className,
}) => {
  return (
    <div className={cn("mb-6", className)}>
      <div className="flex items-center space-x-2">
        {icon && (
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
            {icon}
          </div>
        )}
        <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
      </div>
      {description && (
        <p className="text-muted-foreground mt-1">{description}</p>
      )}
    </div>
  );
};

export default SectionHeader;
