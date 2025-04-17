
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { BarChart2, FileText, Home, Layers, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface NavItem {
  label: string;
  emoji: string;
  icon: React.ReactNode;
  href: string;
}

const navItems: NavItem[] = [
  {
    label: 'Home',
    emoji: '🏠',
    icon: <Home className="h-5 w-5" />,
    href: '/',
  },
  {
    label: 'Preprocess Data',
    emoji: '🧼',
    icon: <Layers className="h-5 w-5" />,
    href: '/preprocess',
  },
  {
    label: 'Train Model',
    emoji: '🛠️',
    icon: <Settings className="h-5 w-5" />,
    href: '/train',
  },
  {
    label: 'Evaluate Model',
    emoji: '📊',
    icon: <BarChart2 className="h-5 w-5" />,
    href: '/evaluate',
  },
];

const AppSidebar = () => {
  const location = useLocation();
  
  return (
    <div className="min-h-screen w-64 bg-sidebar text-sidebar-foreground border-r border-sidebar-border flex flex-col">
      <div className="p-4 border-b border-sidebar-border">
        <h1 className="text-xl font-bold flex items-center">
          <FileText className="mr-2 h-6 w-6 text-mlpurple-400" />
          <span className="text-sidebar-foreground">Data2Model</span>
        </h1>
        <p className="text-xs mt-1 text-sidebar-foreground/70">One Stop Solution to Train Models</p>
      </div>
      
      <nav className="flex-1 p-4">
        <div className="space-y-1">
          {navItems.map((item) => (
            <Button
              key={item.href}
              variant="ghost"
              className={cn(
                "w-full justify-start text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent",
                location.pathname === item.href && 
                "bg-sidebar-accent text-sidebar-foreground"
              )}
              asChild
            >
              <Link to={item.href} className="flex items-center space-x-3">
                <span className="flex items-center">
                  {item.icon}
                </span>
                <span>{item.emoji} {item.label}</span>
              </Link>
            </Button>
          ))}
        </div>
      </nav>
      
      <div className="p-4 border-t border-sidebar-border">
        <div className="rounded-lg bg-sidebar-accent p-3 text-xs">
          <p className="font-medium mb-1 text-sidebar-foreground">Welcome to Data2Model!</p>
          <p className="text-sidebar-foreground/70">
            Follow the workflow to process data, train ML models, and evaluate results.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AppSidebar;
