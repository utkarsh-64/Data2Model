import React, { useState, useRef } from 'react';
import { Upload, FileUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardTitle, CardDescription } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface UploadStepProps {
  onDataLoaded: (metadata: { rows: number; columns: number; preview: any[] }) => void;
}

const UploadStep: React.FC<UploadStepProps> = ({ onDataLoaded }) => {
  const { toast } = useToast();
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (file: File) => {
    if (!file) return;

    const validTypes = ['.csv', '.xlsx', '.xls', 'text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    if (!validTypes.some(type => file.name.endsWith(type) || file.type.includes(type))) {
      setFileError('Please upload a CSV or Excel file');
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setFileError('File size exceeds the 100MB limit');
      return;
    }

    setFileError(null);
    toast({
      title: "Uploading...",
      description: `Sending ${file.name} to backend`,
      className: "bg-blue-50 border-blue-200 dark:bg-blue-950/30 dark:border-blue-900/50",
    });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch("https://data2model.onrender.com/upload-data", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: formData,
        credentials: "include", // ✅ This enables session cookie
      });
      

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');

      toast({
        title: "Data Loaded Successfully",
        description: `Loaded ${data.rows} records and ${data.columns} columns.`,
        className: "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-900/50",
      });
      onDataLoaded({ rows: data.rows, columns: data.columns, preview: data.preview });

    } catch (err: any) {
      toast({
        title: "Upload Failed",
        description: err.message,
        variant: "destructive",
      });
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const openFileSelector = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleLoadSampleData = () => {
    setFileError(null);
    setTimeout(() => {
      onDataLoaded({
        rows: 1245,
        columns: 7,
        preview: [],
      });
      toast({
        title: "Sample Data Loaded Successfully",
        description: "Loaded sample dataset with 1,245 records and 7 columns.",
        className: "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-900/50",
      });
    }, 1000);
  };

  return (
    <Card className={`p-8 text-center border border-dashed transition-all ${isDragging ? 'border-primary bg-primary/5' : ''}`}>
      <div 
        className="max-w-md mx-auto"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="mb-6 bg-muted/40 p-6 rounded-full w-20 h-20 mx-auto flex items-center justify-center">
          <Upload className="h-10 w-10 text-muted-foreground" />
        </div>
        <CardTitle className="mb-2">Upload or Load Your Dataset</CardTitle>
        <CardDescription className="mb-6">
          Start by uploading your data to begin the preprocessing workflow. 
          We support CSV and Excel files.
        </CardDescription>

        <div className="space-y-4">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleInputChange} 
            className="hidden" 
            accept=".csv,.xlsx,.xls"
          />

          <Button onClick={openFileSelector} className="mb-2">
            <FileUp className="mr-2 h-4 w-4" />
            Upload from your computer
          </Button>

          <div className="flex items-center justify-center">
            <div className="border-t flex-1 border-border"></div>
            <span className="px-3 text-xs text-muted-foreground">OR</span>
            <div className="border-t flex-1 border-border"></div>
          </div>

          <Button onClick={handleLoadSampleData} variant="outline" className="mb-4">
            <Upload className="mr-2 h-4 w-4" />
            Load Sample Dataset
          </Button>

          {fileError && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{fileError}</AlertDescription>
            </Alert>
          )}

          <div className="text-xs text-muted-foreground mt-4">
            <p>Drag and drop files here or click to browse</p>
            <p className="mt-1">Maximum file size: 100MB</p>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default UploadStep;
