
import React from 'react';
import { Search, Brain, ChartBar } from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import ProcessingStep from '@/components/ProcessingStep';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="space-y-10 max-w-7xl mx-auto animate-fade-in">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          Data2Model
          <span className="ml-2 bg-clip-text text-transparent bg-gradient-to-r from-mlpurple-500 to-mlblue-500">
            AI Platform
          </span>
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
          Transform your data into powerful machine learning models with our
          all-in-one platform. Preprocess, train, and evaluate in one place.
        </p>
        <div className="pt-4">
          <Button asChild className="bg-gradient-to-r from-mlpurple-500 to-mlpurple-700 hover:from-mlpurple-600 hover:to-mlpurple-800">
            <Link to="/preprocess">
              Get Started
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="rounded-xl overflow-hidden shadow-lg">
          <img 
            src="https://images.unsplash.com/photo-1488590528505-98d2b5aba04b" 
            alt="Data visualization" 
            className="w-full h-64 object-cover" 
          />
          <div className="p-4 bg-white dark:bg-gray-800">
            <p className="font-medium">Interactive data analysis and preprocessing</p>
          </div>
        </div>
        <div className="rounded-xl overflow-hidden shadow-lg">
          <img 
            src="https://images.unsplash.com/photo-1581091226825-a6a2a5aee158" 
            alt="Machine learning concepts" 
            className="w-full h-64 object-cover" 
          />
          <div className="p-4 bg-white dark:bg-gray-800">
            <p className="font-medium">Intelligent model training and evaluation</p>
          </div>
        </div>
      </div>

      <div>
        <SectionHeader 
          title="Three Simple Steps" 
          description="Our streamlined workflow makes machine learning accessible to everyone"
        />
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <ProcessingStep 
            number={1}
            title="Preprocess Data"
            description="Clean and transform your data for optimal model performance"
            icon={<Search className="h-6 w-6" />}
          />
          <ProcessingStep 
            number={2}
            title="Train Model"
            description="Build and optimize machine learning models with ease"
            icon={<Brain className="h-6 w-6" />}
          />
          <ProcessingStep 
            number={3}
            title="Evaluate Results"
            description="Assess model performance with comprehensive metrics"
            icon={<ChartBar className="h-6 w-6" />}
          />
        </div>
      </div>

      <div className="bg-gradient-to-r from-mlpurple-50 to-mlblue-50 dark:from-mlpurple-950/30 dark:to-mlblue-950/30 rounded-xl p-8 shadow-sm">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-2xl font-bold mb-4">Ready to transform your data?</h2>
          <p className="text-muted-foreground mb-6">
            Start your machine learning journey with our comprehensive platform.
            From preprocessing to evaluation, we've got you covered.
          </p>
          <Button asChild className="bg-mlpurple-600 hover:bg-mlpurple-700">
            <Link to="/preprocess">
              Start Preprocessing
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Home;
