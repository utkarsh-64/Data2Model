import React, { useEffect, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { RefreshCw, FileDown, Upload } from "lucide-react";

const TrainModel = () => {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [columns, setColumns] = useState<string[]>([]);
  const [modelType, setModelType] = useState("classification");
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [testSize, setTestSize] = useState(20);
  const [cvFolds, setCvFolds] = useState(5);
  const [scaling, setScaling] = useState(true);
  const [tuneHyperparams, setTuneHyperparams] = useState(false);
  const [enableExport, setEnableExport] = useState(true);
  const [zipReady, setZipReady] = useState(false);
  const [zipFile, setZipFile] = useState("");

  const handleFileUpload = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("https://data2model.onrender.com/upload-data", {
        method: "POST",
        credentials: "include",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed");
      setUploaded(true);
      toast({ title: "Upload Successful", description: data.message });
      // Fetch column info
      const edaRes = await fetch("https://data2model.onrender.com/eda", {
        method: "POST",
        credentials: "include",
      });
      const edaData = await edaRes.json();
      if (edaData.columns) setColumns(edaData.columns.map((c: any) => c.name));
    } catch (err: any) {
      toast({ title: "Upload Failed", description: err.message, variant: "destructive" });
    } finally {
      setUploading(false);
    }
  };

  const algorithmOptions = modelType === "classification"
    ? ["knn", "random_forest", "logistic_regression", "decision_trees", "naive_bayes", "sgd", "svm"]
    : ["knn", "linear", "ridge", "lasso", "decision_trees", "random_forest", "svm"];

  const toggleAlgorithm = (alg: string) => {
    setSelectedAlgorithms((prev) =>
      prev.includes(alg) ? prev.filter((a) => a !== alg) : [...prev, alg]
    );
  };

  const handleTrain = async () => {
    if (!uploaded || !selectedAlgorithms.length || !targetColumn) {
      toast({ title: "Missing info", description: "Upload data and select all options", variant: "destructive" });
      return;
    }
    try {
      const res = await fetch("https://data2model.onrender.com/train-model", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_type: modelType,
          algorithms: selectedAlgorithms,
          target: targetColumn,
          test_size: testSize / 100,
          cv_folds: cvFolds,
          scale: scaling,
          tune_hyperparams: tuneHyperparams,
          export_zip: enableExport,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      if (data.zip_path) {
        setZipFile(data.zip_path);
        setZipReady(true);
        toast({ title: "Training & Export Complete", description: "ZIP file is ready for download." });
      }
    } catch (err: any) {
      toast({ title: "Training failed", description: err.message, variant: "destructive" });
    }
  };

  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Train & Export Models</CardTitle>
        <CardDescription>Upload data, configure and export models as a ZIP</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center gap-4">
          <Input type="file" accept=".csv,.xlsx" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <Button onClick={handleFileUpload} disabled={!file || uploading}>
            <Upload className="mr-2 h-4 w-4" /> Upload
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label>Model Type</Label>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger><SelectValue placeholder="Select type" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="classification">Classification</SelectItem>
                <SelectItem value="regression">Regression</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label>Target Column</Label>
            <Select value={targetColumn} onValueChange={setTargetColumn}>
              <SelectTrigger><SelectValue placeholder="Select target" /></SelectTrigger>
              <SelectContent>
                {columns.map(col => <SelectItem key={col} value={col}>{col}</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div>
          <Label>Algorithms</Label>
          <div className="flex flex-wrap gap-2 mt-2">
            {algorithmOptions.map(alg => (
              <Button
                key={alg}
                variant={selectedAlgorithms.includes(alg) ? "default" : "outline"}
                onClick={() => toggleAlgorithm(alg)}
              >{alg}</Button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label>Test Size: {testSize}%</Label>
            <Slider min={10} max={50} step={5} value={[testSize]} onValueChange={(v) => setTestSize(v[0])} />
          </div>
          <div>
            <Label>CV Folds</Label>
            <Input type="number" value={cvFolds} onChange={(e) => setCvFolds(Number(e.target.value))} />
          </div>
        </div>

        <div className="flex gap-4 mt-4">
          <Label className="flex items-center gap-2">
            <Switch checked={scaling} onCheckedChange={setScaling} /> Scaling
          </Label>
          <Label className="flex items-center gap-2">
            <Switch checked={tuneHyperparams} onCheckedChange={setTuneHyperparams} /> Hyperparameter Tuning
          </Label>
          <Label className="flex items-center gap-2">
            <Switch checked={enableExport} onCheckedChange={setEnableExport} /> Enable Export
          </Label>
        </div>

        <Button onClick={handleTrain} disabled={!uploaded || !selectedAlgorithms.length || !targetColumn}>
          {uploading ? <RefreshCw className="mr-2 h-4 w-4 animate-spin" /> : "Start Training"}
        </Button>

        {zipReady && (
          <div className="mt-4">
            <a href={`https://data2model.onrender.com${zipFile}`} download>
              <Button className="w-full" variant="secondary">
                <FileDown className="mr-2 h-4 w-4" /> Download Model ZIP
              </Button>
            </a>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TrainModel;
