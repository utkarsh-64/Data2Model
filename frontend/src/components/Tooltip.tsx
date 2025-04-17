
import React from 'react';
import { cn } from '@/lib/utils';
import { Info } from 'lucide-react';

interface TooltipProps {
  content: string;
  children: React.ReactNode;
  className?: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  icon?: boolean;
}

const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  className,
  position = 'top',
  icon = false,
}) => {
  const getPositionClasses = () => {
    switch (position) {
      case 'bottom':
        return 'top-full mt-2 bottom-auto mb-0';
      case 'left':
        return 'right-full mr-2 left-auto ml-0 bottom-auto top-1/2 -translate-y-1/2 translate-x-0';
      case 'right':
        return 'left-full ml-2 right-auto mr-0 bottom-auto top-1/2 -translate-y-1/2 translate-x-0';
      case 'top':
      default:
        return 'bottom-full mb-2 top-auto mt-0';
    }
  };

  return (
    <div className={cn("tooltip-wrapper", className)}>
      {icon ? (
        <span className="cursor-help text-muted-foreground hover:text-foreground transition-colors">
          <Info className="h-4 w-4" />
        </span>
      ) : (
        children
      )}
      <div className={cn("tooltip-content", getPositionClasses())}>
        {content}
      </div>
    </div>
  );
};

export default Tooltip;
