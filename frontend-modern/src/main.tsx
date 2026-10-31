import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@radix-ui/react-tooltip";
import "./styles.css";

const App = lazy(() => import("./App").then(module => ({default: module.App})));

const client = new QueryClient({defaultOptions:{queries:{staleTime:30_000,retry:1,refetchOnWindowFocus:false}}});
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><QueryClientProvider client={client}><TooltipProvider delayDuration={300}><Suspense fallback={<div className="app-boot">Opening ASTRA…</div>}><App /></Suspense></TooltipProvider></QueryClientProvider></React.StrictMode>
);
