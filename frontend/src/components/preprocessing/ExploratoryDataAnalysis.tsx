import React, { useEffect, useState } from 'react';
import { Search, Filter, BarChart2, LineChart, ArrowRight } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';

interface ColumnStats {
  name: string;
  type: string;
  missing: number;
  unique: number;
  sample_values: string[];
}

interface SummaryStats {
  total_records: number;
  total_features: number;
  numeric_count: number;
  categorical_count: number;
}

const ExploratoryDataAnalysis = ({ onNext }: { onNext: () => void }) => {
  const { toast } = useToast();
  const [columns, setColumns] = useState<ColumnStats[]>([]);
  const [summary, setSummary] = useState<SummaryStats | null>(null);
  const [plotURL, setPlotURL] = useState<string>('');
  const [missingURL, setMissingURL] = useState<string>('');
  const [correlationURL, setCorrelationURL] = useState<string>('');

  useEffect(() => {
    fetch("https://data2model.onrender.com/eda", { method: "POST", credentials: "include" })
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        setColumns(data.columns);
        setSummary(data.summary);
      })
      .catch(err => {
        toast({
          title: "EDA Load Failed",
          description: err.message,
          variant: "destructive",
        });
      });

    fetch("https://data2model.onrender.com/eda/distribution", { method: "GET", credentials: "include" })
      .then(res => res.blob())
      .then(blob => setPlotURL(URL.createObjectURL(blob)));

    fetch("https://data2model.onrender.com/eda/missing", { method: "GET", credentials: "include" })
      .then(res => res.blob())
      .then(blob => setMissingURL(URL.createObjectURL(blob)));

    fetch("https://data2model.onrender.com/eda/correlation", { method: "GET", credentials: "include" })
      .then(res => res.blob())
      .then(blob => setCorrelationURL(URL.createObjectURL(blob)));
  }, []);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Exploratory Data Analysis</CardTitle>
        <CardDescription>
          Understand your data structure, distributions, and relationships
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <Tabs defaultValue="overview">
          <TabsList className="w-full justify-start mb-6">
            <TabsTrigger value="overview"><Search className="mr-2 h-4 w-4" /> Overview</TabsTrigger>
            <TabsTrigger value="distribution"><BarChart2 className="mr-2 h-4 w-4" /> Distribution</TabsTrigger>
            <TabsTrigger value="correlations"><LineChart className="mr-2 h-4 w-4" /> Correlation</TabsTrigger>
            <TabsTrigger value="missing"><Filter className="mr-2 h-4 w-4" /> Missing</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <div>
              <h3 className="text-lg font-medium mb-4">Data Summary</h3>
              <div className="overflow-auto rounded-lg border">
                <table className="min-w-full divide-y divide-border">
                  <thead className="bg-muted/40">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Column</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Type</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Missing</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Unique</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">Sample Values</th>
                    </tr>
                  </thead>
                  <tbody className="bg-background divide-y divide-border">
                    {columns.map((col, index) => (
                      <tr key={index}>
                        <td className="px-4 py-3 whitespace-nowrap text-sm font-medium">{col.name}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm">{col.type}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm">{(col.missing * 100).toFixed(1)}%</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm">{col.unique}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm truncate max-w-xs">{col.sample_values.join(", ")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {summary && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-muted/20 border rounded-lg p-4">
                  <div className="text-sm text-muted-foreground mb-1">Total Records</div>
                  <div className="text-2xl font-bold text-mlpurple-600 dark:text-mlpurple-400">{summary.total_records}</div>
                </div>
                <div className="bg-muted/20 border rounded-lg p-4">
                  <div className="text-sm text-muted-foreground mb-1">Total Features</div>
                  <div className="text-2xl font-bold text-mlpurple-600 dark:text-mlpurple-400">{summary.total_features}</div>
                </div>
                <div className="bg-muted/20 border rounded-lg p-4">
                  <div className="text-sm text-muted-foreground mb-1">Numeric Features</div>
                  <div className="text-2xl font-bold text-mlpurple-600 dark:text-mlpurple-400">{summary.numeric_count}</div>
                </div>
                <div className="bg-muted/20 border rounded-lg p-4">
                  <div className="text-sm text-muted-foreground mb-1">Categorical Features</div>
                  <div className="text-2xl font-bold text-mlpurple-600 dark:text-mlpurple-400">{summary.categorical_count}</div>
                </div>
              </div>
            )}
          </TabsContent>

          <TabsContent value="distribution">
            {plotURL ? (
              <img src={plotURL} alt="Distribution Plot" className="rounded-lg border" />
            ) : <p className="text-muted-foreground">Loading distribution...</p>}
          </TabsContent>

          <TabsContent value="correlations">
            {correlationURL ? (
              <img src={correlationURL} alt="Correlation Heatmap" className="rounded-lg border" />
            ) : <p className="text-muted-foreground">Loading correlation...</p>}
          </TabsContent>

          <TabsContent value="missing">
            {missingURL ? (
              <img src={missingURL} alt="Missing Values Matrix" className="rounded-lg border" />
            ) : <p className="text-muted-foreground">Loading missing values plot...</p>}
          </TabsContent>
        </Tabs>

        <div className="flex justify-end">
          <Button onClick={onNext}>
            Continue to Data Cleaning
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ExploratoryDataAnalysis;