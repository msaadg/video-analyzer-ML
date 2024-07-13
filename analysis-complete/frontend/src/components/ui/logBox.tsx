export const LogBox = ({ label, logs } : { label: string, logs: string[] }) => {
  return (
    <div className="mb-8">
      <label htmlFor="message" className="block mb-2 text-md font-medium text-gray-900">{label}</label>
      <textarea id="message" rows={8} className="block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 shadow-lg" placeholder="" value={logs.join('\n')} readOnly></textarea>
    </div>
  );
};