
import React from 'react';
import { cn } from '@/lib/utils';
import { 
  CheckCircle, 
  AlertCircle, 
  Info, 
  AlertTriangle, 
  LoaderCircle
} from 'lucide-react';

type StatusType = 'success' | 'error' | 'warning' | 'info' | 'loading';

interface StatusBadgeProps {
  status: StatusType;
  text: string;
  className?: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ 
  status, 
  text,
  className 
}) => {
  const getStatusClasses = () => {
    switch (status) {
      case 'success':
        return 'bg-green-50 text-green-700 border-green-200 dark:bg-green-950/30 dark:text-green-400 dark:border-green-900/50';
      case 'error':
        return 'bg-red-50 text-red-700 border-red-200 dark:bg-red-950/30 dark:text-red-400 dark:border-red-900/50';
      case 'warning':
        return 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/30 dark:text-amber-400 dark:border-amber-900/50';
      case 'info':
        return 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:text-blue-400 dark:border-blue-900/50';
      case 'loading':
        return 'bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-800/50 dark:text-gray-300 dark:border-gray-700';
      default:
        return 'bg-gray-50 text-gray-700 border-gray-200 dark:bg-gray-800/50 dark:text-gray-300 dark:border-gray-700';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4" />;
      case 'error':
        return <AlertCircle className="h-4 w-4" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4" />;
      case 'info':
        return <Info className="h-4 w-4" />;
      case 'loading':
        return <LoaderCircle className="h-4 w-4 animate-spin" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  return (
    <div className={cn(
      "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border",
      getStatusClasses(),
      className
    )}>
      {getStatusIcon()}
      <span>{text}</span>
    </div>
  );
};

export default StatusBadge;
