
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import PageLayout from "@/components/PageLayout";
import Home from "@/pages/Home";
import PreprocessData from "@/pages/PreprocessData";
import TrainModel from "@/pages/TrainModel";
import EvaluateModel from "@/pages/EvaluateModel";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route 
            path="/" 
            element={
              <PageLayout>
                <Home />
              </PageLayout>
            } 
          />
          <Route 
            path="/preprocess" 
            element={
              <PageLayout>
                <PreprocessData />
              </PageLayout>
            } 
          />
          <Route 
            path="/train" 
            element={
              <PageLayout>
                <TrainModel />
              </PageLayout>
            } 
          />
          <Route 
            path="/evaluate" 
            element={
              <PageLayout>
                <EvaluateModel />
              </PageLayout>
            } 
          />
          <Route 
            path="*" 
            element={
              <PageLayout>
                <NotFound />
              </PageLayout>
            } 
          />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
