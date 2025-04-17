import React, { useEffect, useState } from 'react';
import {
  ArrowRight,
  Columns,
  Filter,
  Edit,
  FileUp
} from 'lucide-react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from '@/components/ui/collapsible';
import { toast } from '@/components/ui/use-toast';

interface FeatureEngineeringProps {
  onNext: () => void;
}

interface ColumnInfo {
  name: string;
  type: 'numeric' | 'categorical';
}

const FeatureEngineering: React.FC<FeatureEngineeringProps> = ({ onNext }) => {
  const [columns, setColumns] = useState<ColumnInfo[]>([]);
  const [newColumn, setNewColumn] = useState('');
  const [expression, setExpression] = useState('');
  const [scalingMethod, setScalingMethod] = useState('');
  const [scaleCols, setScaleCols] = useState<Set<string>>(new Set());
  const [encodingMethod, setEncodingMethod] = useState('');
  const [encodeCols, setEncodeCols] = useState<Set<string>>(new Set());
  const [binCol, setBinCol] = useState('');
  const [binCount, setBinCount] = useState(5);
  const [binStrategy, setBinStrategy] = useState('');

  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const res = await fetch('http://localhost:5000/eda', {
          method: 'POST',
          credentials: 'include'
        });
        const data = await res.json();
        if (data.columns) {
          setColumns(data.columns);
        }
      } catch (err) {
        toast({
          title: 'Error',
          description: 'Failed to load columns',
          variant: 'destructive'
        });
      }
    };
    fetchColumns();
  }, []);

  const handleCreateFeature = async () => {
    try {
      const res = await fetch('http://localhost:5000/create-feature', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          column: newColumn,
          expression
        })
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error);
      toast({ title: 'Feature Created', description: result.message });
    } catch (err: any) {
      toast({
        title: 'Failed to create feature',
        description: err.message,
        variant: 'destructive'
      });
    }
  };

  const handleApplyScaling = async () => {
    try {
      const res = await fetch('http://localhost:5000/scale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          method: scalingMethod,
          columns: Array.from(scaleCols)
        })
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error);
      toast({ title: 'Scaling Applied', description: result.message });
    } catch (err: any) {
      toast({
        title: 'Scaling Failed',
        description: err.message,
        variant: 'destructive'
      });
    }
  };

  const handleEncoding = async () => {
    try {
      const res = await fetch('http://localhost:5000/encode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          method: encodingMethod,
          columns: Array.from(encodeCols)
        })
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error);
      toast({ title: 'Encoding Applied', description: result.message });
    } catch (err: any) {
      toast({
        title: 'Encoding Failed',
        description: err.message,
        variant: 'destructive'
      });
    }
  };

  const handleBinning = async () => {
    try {
      const res = await fetch('http://localhost:5000/bin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          column: binCol,
          bins: binCount,
          strategy: binStrategy
        })
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.error);
      toast({ title: 'Binning Applied', description: result.message });
    } catch (err: any) {
      toast({
        title: 'Binning Failed',
        description: err.message,
        variant: 'destructive'
      });
    }
  };

  const numericCols = columns.filter(col => col.type === 'numeric');
  const categoricalCols = columns.filter(col => col.type === 'categorical');

  return (
    <Card>
      <CardHeader>
        <CardTitle>Feature Engineering</CardTitle>
        <CardDescription>
          Transform existing features and create new ones to improve model performance
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Create Feature */}
          <Collapsible className="border rounded-lg p-4">
            <CollapsibleTrigger className="flex justify-between items-center w-full">
              <div className="flex items-center">
                <div className="bg-primary/10 p-2 rounded-full mr-3">
                  <Columns className="h-5 w-5 text-primary" />
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-medium">Create New Features</h3>
                  <p className="text-sm text-muted-foreground">
                    Derive new columns from existing data
                  </p>
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4 space-y-4">
              <Label htmlFor="new-column">New Column Name</Label>
              <Input
                id="new-column"
                value={newColumn}
                onChange={(e) => setNewColumn(e.target.value)}
              />
              <Label htmlFor="column-expression">Expression</Label>
              <Input
                id="column-expression"
                value={expression}
                onChange={(e) => setExpression(e.target.value)}
              />
              <Button onClick={handleCreateFeature}>Create Feature</Button>
            </CollapsibleContent>
          </Collapsible>

          {/* Feature Scaling */}
          <Collapsible className="border rounded-lg p-4">
            <CollapsibleTrigger className="flex justify-between items-center w-full">
              <div className="flex items-center">
                <div className="bg-primary/10 p-2 rounded-full mr-3">
                  <Filter className="h-5 w-5 text-primary" />
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-medium">Feature Scaling</h3>
                  <p className="text-sm text-muted-foreground">Normalize or standardize numeric features</p>
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4 space-y-4">
              <Label htmlFor="scaling-method">Scaling Method</Label>
              <Select onValueChange={setScalingMethod}>
                <SelectTrigger id="scaling-method">
                  <SelectValue placeholder="Select method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="minmax">Min-Max</SelectItem>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="robust">Robust</SelectItem>
                  <SelectItem value="log">Log</SelectItem>
                </SelectContent>
              </Select>
              <Label>Columns to Scale</Label>
              <div className="grid grid-cols-2 gap-2">
                {numericCols.map((col) => (
                  <div key={col.name} className="flex items-center space-x-2">
                    <Checkbox
                      checked={scaleCols.has(col.name)}
                      onCheckedChange={(checked) => {
                        setScaleCols((prev) => {
                          const set = new Set(prev);
                          checked ? set.add(col.name) : set.delete(col.name);
                          return set;
                        });
                      }}
                    />
                    <Label className="text-sm">{col.name}</Label>
                  </div>
                ))}
              </div>
              <Button onClick={handleApplyScaling}>Apply Scaling</Button>
            </CollapsibleContent>
          </Collapsible>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Encoding */}
          <Collapsible className="border rounded-lg p-4">
            <CollapsibleTrigger className="flex justify-between items-center w-full">
              <div className="flex items-center">
                <div className="bg-primary/10 p-2 rounded-full mr-3">
                  <Edit className="h-5 w-5 text-primary" />
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-medium">Encoding Categorical Features</h3>
                  <p className="text-sm text-muted-foreground">
                    Convert categorical data to numeric
                  </p>
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4 space-y-4">
              <Label htmlFor="encoding-method">Encoding Method</Label>
              <Select onValueChange={setEncodingMethod}>
                <SelectTrigger id="encoding-method">
                  <SelectValue placeholder="Select method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="onehot">One-Hot</SelectItem>
                  <SelectItem value="label">Label</SelectItem>
                  <SelectItem value="target">Target</SelectItem>
                </SelectContent>
              </Select>
              <Label>Columns to Encode</Label>
              <div className="grid grid-cols-2 gap-2">
                {categoricalCols.map((col) => (
                  <div key={col.name} className="flex items-center space-x-2">
                    <Checkbox
                      checked={encodeCols.has(col.name)}
                      onCheckedChange={(checked) => {
                        setEncodeCols((prev) => {
                          const set = new Set(prev);
                          checked ? set.add(col.name) : set.delete(col.name);
                          return set;
                        });
                      }}
                    />
                    <Label className="text-sm">{col.name}</Label>
                  </div>
                ))}
              </div>
              <Button onClick={handleEncoding}>Apply Encoding</Button>
            </CollapsibleContent>
          </Collapsible>

          {/* Binning */}
          <Collapsible className="border rounded-lg p-4">
            <CollapsibleTrigger className="flex justify-between items-center w-full">
              <div className="flex items-center">
                <div className="bg-primary/10 p-2 rounded-full mr-3">
                  <FileUp className="h-5 w-5 text-primary" />
                </div>
                <div className="text-left">
                  <h3 className="text-lg font-medium">Binning Numeric Features</h3>
                  <p className="text-sm text-muted-foreground">
                    Group continuous values into discrete bins
                  </p>
                </div>
              </div>
              <ArrowRight className="h-4 w-4" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4 space-y-4">
              <Label htmlFor="binning-column">Column to Bin</Label>
              <Select onValueChange={setBinCol}>
                <SelectTrigger id="binning-column">
                  <SelectValue placeholder="Select column" />
                </SelectTrigger>
                <SelectContent>
                  {numericCols.map(col => (
                    <SelectItem key={col.name} value={col.name}>{col.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Label htmlFor="bin-count">Number of Bins</Label>
              <Input
                id="bin-count"
                type="number"
                min={2}
                max={20}
                value={binCount}
                onChange={(e) => setBinCount(Number(e.target.value))}
              />
              <Label htmlFor="bin-strategy">Binning Strategy</Label>
              <Select onValueChange={setBinStrategy}>
                <SelectTrigger id="bin-strategy">
                  <SelectValue placeholder="Select strategy" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="uniform">Uniform</SelectItem>
                  <SelectItem value="quantile">Quantile</SelectItem>
                  <SelectItem value="kmeans">K-means</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleBinning}>Apply Binning</Button>
            </CollapsibleContent>
          </Collapsible>
        </div>

        <div className="flex justify-end">
          <Button onClick={onNext}>
            Continue to Export
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default FeatureEngineering;
