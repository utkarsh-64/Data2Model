
import React from 'react';
import StatusBadge from '@/components/StatusBadge';
import { X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface DataSummaryProps {
  onReset: () => void;
}

const DataSummary: React.FC<DataSummaryProps> = ({ onReset }) => {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <StatusBadge status="success" text="Data Loaded" />
        <span className="text-sm text-muted-foreground">1,245 records • 7 columns</span>
      </div>
      <Button variant="outline" size="sm" onClick={onReset}>
        <X className="h-4 w-4 mr-2" />
        Reset
      </Button>
    </div>
  );
};

export default DataSummary;
