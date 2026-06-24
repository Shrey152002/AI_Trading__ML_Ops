import { AlertTriangle } from "lucide-react";

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
      <p className="flex items-center gap-1.5 font-medium">
        <AlertTriangle className="h-4 w-4" />
        Something went wrong
      </p>
      <p className="mt-1 text-red-700">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 rounded-md border border-red-300 bg-white px-3 py-1.5 text-xs font-medium text-red-800 hover:bg-red-100"
        >
          Retry
        </button>
      )}
    </div>
  );
}
