import React, { useState } from 'react';
import { ArrowLeft, Download, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';

interface ExportDataProps {
  onPrevious: () => void;
}

const ExportData: React.FC<ExportDataProps> = ({ onPrevious }) => {
  const { toast } = useToast();
  const [isExporting, setIsExporting] = useState(false);
  const [exportComplete, setExportComplete] = useState(false);

  const handleExport = async () => {
    setIsExporting(true);

    try {
      // Make a request to the backend to generate and return the CSV
      const res = await fetch('http://localhost:5000/export', { 
        method: 'GET' ,
        credentials: "include"
      });
      
      if (!res.ok) {
        throw new Error('Failed to export the data');
      }

      // Get the file as a blob
      const blob = await res.blob();

      // Create a download link and trigger the download
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'processed_dataset.csv';  // Adjust the file name as needed
      document.body.appendChild(link);
      link.click();
      link.remove();

      // Notify the user that the export is complete
      setExportComplete(true);
      toast({
        title: "Export Complete",
        description: "Your processed dataset has been exported as CSV successfully.",
        className: "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-900/50",
      });
    } catch (error: any) {
      // Handle errors
      toast({
        title: "Export Failed",
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Export Processed Data</CardTitle>
        <CardDescription>Download your clean and preprocessed dataset as CSV</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Data summary and processing steps */}
        </div>

        {exportComplete && (
          <div className="p-4 rounded-md bg-green-50 border border-green-200 dark:bg-green-950/30 dark:border-green-800/50">
            <div className="flex items-start space-x-3">
              <CheckCircle2 className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-green-800 dark:text-green-300">
                  Export Completed Successfully
                </p>
                <p className="text-xs text-green-700 dark:text-green-400 mt-1">
                  Your processed dataset has been exported as CSV and is ready for model training.
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between border-t px-6 py-4">
        <Button variant="outline" onClick={onPrevious}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Feature Engineering
        </Button>

        <Button onClick={handleExport} disabled={isExporting || exportComplete}>
          {isExporting ? (
            <>
              <span className="mr-2 h-4 w-4 animate-spin">⏳</span>
              Exporting...
            </>
          ) : exportComplete ? (
            <>
              <CheckCircle2 className="mr-2 h-4 w-4" />
              Exported
            </>
          ) : (
            <>
              <Download className="mr-2 h-4 w-4" />
              Export as CSV
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ExportData;
