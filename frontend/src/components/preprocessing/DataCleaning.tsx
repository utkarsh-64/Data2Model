import React, { useEffect, useState } from 'react';
import { ArrowRight, Trash2, Edit } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Checkbox } from '@/components/ui/checkbox';
import { toast } from '@/components/ui/use-toast';

interface DataCleaningProps {
  onNext: () => void;
}

const DataCleaning: React.FC<DataCleaningProps> = ({ onNext }) => {
  const [columns, setColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMethods, setSelectedMethods] = useState<Record<string, string>>({});
  const [selectedDrops, setSelectedDrops] = useState<Set<string>>(new Set());
  const [replaceCol, setReplaceCol] = useState('');
  const [replaceFrom, setReplaceFrom] = useState('');
  const [replaceTo, setReplaceTo] = useState('');

  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const res = await fetch('https://data2model.onrender.com/eda', {
          method: 'POST',
          credentials: 'include',
        });
        const data = await res.json();
        if (data.columns) {
          setColumns(data.columns.map((col: any) => col.name));
        }
      } catch (err) {
        toast({
          title: 'Error loading columns',
          description: 'Make sure you uploaded the data file.',
          variant: 'destructive',
        });
      } finally {
        setLoading(false);
      }
    };
    fetchColumns();
  }, []);

  const handleApplyImputation = async (column: string, method: string, customValue?: string) => {
    try {
      const res = await fetch('https://data2model.onrender.com/clean-data', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ column, method, customValue }),
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error || 'Cleaning failed');
      toast({ title: 'Column Cleaned', description: `Applied ${method} to ${column}` });
    } catch (err: any) {
      toast({ title: 'Cleaning Failed', description: err.message, variant: 'destructive' });
    }
  };

  const handleDropColumns = async () => {
    try {
      const res = await fetch('https://data2model.onrender.com/drop-columns', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ columns: Array.from(selectedDrops) }),
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error || 'Drop failed');
      toast({ title: 'Dropped Columns', description: result.message });
      setColumns(columns.filter(col => !selectedDrops.has(col)));
      setSelectedDrops(new Set());
    } catch (err: any) {
      toast({ title: 'Drop Failed', description: err.message, variant: 'destructive' });
    }
  };

  const handleReplaceValues = async () => {
    try {
      const res = await fetch('https://data2model.onrender.com/replace-values', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ column: replaceCol, find: replaceFrom, replace: replaceTo }),
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error || 'Replace failed');
      toast({ title: 'Value Replaced', description: result.message });
    } catch (err: any) {
      toast({ title: 'Replace Failed', description: err.message, variant: 'destructive' });
    }
  };

  return (
    <Tabs defaultValue="missing">
      <TabsList className="mb-6">
        <TabsTrigger value="missing">Missing Value Treatment</TabsTrigger>
        <TabsTrigger value="column">Column Operations</TabsTrigger>
      </TabsList>
      <TabsContent value="missing" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Missing Value Treatment</CardTitle>
            <CardDescription>
              Handle missing values in your dataset to ensure model accuracy
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium mb-4">Column-wise Imputation</h3>
                <div className="overflow-auto rounded-lg border">
                  <table className="min-w-full divide-y divide-border">
                    <thead className="bg-muted/40">
                      <tr>
                        <th className="px-4 py-3">Column</th>
                        <th className="px-4 py-3">Method</th>
                        <th className="px-4 py-3">Custom Value</th>
                        <th className="px-4 py-3">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {columns.map((column) => (
                        <tr key={column}>
                          <td className="px-4 py-3 text-sm">{column}</td>
                          <td className="px-4 py-3">
                            <Select
                              defaultValue="mean"
                              onValueChange={(val) => {
                                setSelectedMethods((prev) => ({ ...prev, [column]: val }));
                              }}
                            >
                              <SelectTrigger className="w-32">
                                <SelectValue placeholder="Select method" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="mean">Mean</SelectItem>
                                <SelectItem value="median">Median</SelectItem>
                                <SelectItem value="mode">Mode</SelectItem>
                                <SelectItem value="custom">Custom</SelectItem>
                              </SelectContent>
                            </Select>
                          </td>
                          <td className="px-4 py-3">
                            <Input id={`custom-${column}`} type="text" className="w-36" />
                          </td>
                          <td className="px-4 py-3">
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => {
                                const method = selectedMethods[column] || 'mean';
                                const custom = (document.getElementById(`custom-${column}`) as HTMLInputElement)?.value;
                                handleApplyImputation(column, method, custom);
                              }}
                            >
                              <ArrowRight className="h-4 w-4 mr-1" />
                              Apply
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        <div className="flex justify-end">
          <Button onClick={onNext}>
            Continue to Feature Engineering
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </TabsContent>
      <TabsContent value="column" className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Column Operations</CardTitle>
            <CardDescription>
              Modify, transform, and manage your dataset columns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium mb-4">Drop Columns</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {columns.map((column, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <Checkbox
                        id={`drop-${column}`}
                        checked={selectedDrops.has(column)}
                        onCheckedChange={(checked) => {
                          setSelectedDrops((prev) => {
                            const updated = new Set(prev);
                            checked ? updated.add(column) : updated.delete(column);
                            return updated;
                          });
                        }}
                      />
                      <Label htmlFor={`drop-${column}`} className="text-sm">
                        {column}
                      </Label>
                    </div>
                  ))}
                </div>
                <Button variant="outline" className="mt-4" onClick={handleDropColumns}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Drop Selected Columns
                </Button>
              </div>
              <Separator />
              <div>
                <h3 className="text-lg font-medium mb-4">Replace Values</h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <div>
                      <Label htmlFor="replace-column">Column</Label>
                      <Select onValueChange={(val) => setReplaceCol(val)}>
                        <SelectTrigger id="replace-column">
                          <SelectValue placeholder="Select column" />
                        </SelectTrigger>
                        <SelectContent>
                          {columns.map((column, index) => (
                            <SelectItem key={index} value={column}>{column}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="old-value">Find value</Label>
                      <Input id="old-value" placeholder="Value to replace" value={replaceFrom} onChange={(e) => setReplaceFrom(e.target.value)} />
                    </div>
                    <div>
                      <Label htmlFor="new-value">Replace with</Label>
                      <Input id="new-value" placeholder="New value" value={replaceTo} onChange={(e) => setReplaceTo(e.target.value)} />
                    </div>
                    <Button onClick={handleReplaceValues}>
                      <Edit className="h-4 w-4 mr-2" />
                      Replace Values
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        <div className="flex justify-end">
          <Button onClick={onNext}>
            Continue to Feature Engineering
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </TabsContent>
    </Tabs>
  );
};

export default DataCleaning;
