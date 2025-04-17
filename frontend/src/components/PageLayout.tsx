
import React from 'react';
import AppSidebar from './AppSidebar';
import { Toaster } from '@/components/ui/toaster';

interface PageLayoutProps {
  children: React.ReactNode;
}

const PageLayout: React.FC<PageLayoutProps> = ({ children }) => {
  return (
    <div className="flex min-h-screen bg-background">
      <AppSidebar />
      <div className="flex-1 flex flex-col">
        <main className="flex-1 p-6 overflow-auto">
          {children}
        </main>
        <footer className="py-4 px-6 border-t border-border text-center text-sm text-muted-foreground">
          <p>
            Need help? Contact support at{' '}
            <a 
              href="mailto:support@data2model.com" 
              className="text-primary hover:underline"
            >
              support@data2model.com
            </a>
          </p>
        </footer>
      </div>
      <Toaster />
    </div>
  );
};

export default PageLayout;
