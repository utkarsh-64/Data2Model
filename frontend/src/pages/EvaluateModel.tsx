import React, { useState } from 'react';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';

const EvaluateModel = () => {
  const { toast } = useToast();
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [task, setTask] = useState<'classification' | 'regression'>('classification');
  const [evaluationResults, setEvaluationResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleModelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setModelFile(e.target.files[0]);
  };

  const handleTestUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setTestFile(e.target.files[0]);
  };

  const handleEvaluate = async () => {
    if (!modelFile || !testFile) {
      toast({ title: 'Upload Required', description: 'Please upload both model and test dataset.', variant: 'destructive' });
      return;
    }

    const formData = new FormData();
    formData.append('model', modelFile);
    formData.append('test_data', testFile);
    formData.append('task', task);

    setLoading(true);
    try {
      const res = await fetch('http://localhost:5000/evaluate', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Evaluation failed');
      setEvaluationResults(data);
      toast({ title: 'Evaluation Complete', description: 'Results are displayed below.' });
    } catch (err: any) {
      toast({ title: 'Evaluation Failed', description: err.message || 'Unexpected error.', variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-3xl mx-auto py-10">
      <Card>
        <CardHeader>
          <CardTitle>Evaluate Model</CardTitle>
          <CardDescription>Upload a trained model and test dataset (CSV) to view evaluation metrics</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="task-type">Task Type</Label>
            <Select value={task} onValueChange={(value: 'classification' | 'regression') => setTask(value)}>
              <SelectTrigger id="task-type">
                <SelectValue placeholder="Select task type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="classification">Classification</SelectItem>
                <SelectItem value="regression">Regression</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label>Upload Model (.pkl)</Label>
            <Input type="file" accept=".pkl,.joblib" onChange={handleModelUpload} />
          </div>

          <div>
            <Label>Upload Test Data (.csv)</Label>
            <Input type="file" accept=".csv" onChange={handleTestUpload} />
          </div>

          <Button onClick={handleEvaluate} disabled={loading}>
            {loading ? 'Evaluating...' : 'Evaluate Model'}
          </Button>
        </CardContent>
      </Card>

      {/* {evaluationResults && (
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-muted p-4 rounded-lg text-sm whitespace-pre-wrap overflow-x-auto">
              {JSON.stringify(evaluationResults, null, 2)}
            </div>
          </CardContent>
        </Card>
      )} */}
      {
      evaluationResults && ( 
        <Card> 
          <CardHeader> 
            <CardTitle>Evaluation Results</CardTitle> 
          </CardHeader> 
          <CardContent className="space-y-4"> 
            {evaluationResults.task === 'regression' ? ( <div className="space-y-2"> <p><strong>MAE:</strong> {evaluationResults.metrics.MAE}</p> <p><strong>MSE:</strong> {evaluationResults.metrics.MSE}</p> <p><strong>RMSE:</strong> {evaluationResults.metrics.RMSE}</p> <p><strong>R² Score:</strong> {evaluationResults.metrics.R2}</p> </div> ) : ( <div className="space-y-2"> <h4 className="font-semibold">Classification Report</h4> <pre className="text-sm bg-muted p-2 rounded-lg overflow-auto"> {JSON.stringify(evaluationResults.report, null, 2)} </pre> <h4 className="font-semibold">Confusion Matrix</h4> <table className="text-sm border"> <thead> <tr> <th className="p-2">Actual \ Pred</th> {evaluationResults.labels.map((label: string | number, i: number) => ( <th key={i} className="p-2">{label}</th> ))} </tr> </thead> <tbody> {evaluationResults.confusion_matrix.map((row: number[], i: number) => ( <tr key={i}> <td className="p-2 font-medium">{evaluationResults.labels[i]}</td> {row.map((val: number, j: number) => ( <td key={j} className="p-2 text-center">{val}</td> ))} </tr> ))} </tbody> </table> </div> )} 
          </CardContent> 
        </Card> 
      )}
    </div>
  );
};

export default EvaluateModel;
