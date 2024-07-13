import axios from "axios";

interface AnalyzeButtonProps {
  videoUploaded: boolean;
  setAnalysisStarted: (started: boolean) => void;
  uploadedFilePath: string;
}

export const AnalyzeButton: React.FC<AnalyzeButtonProps> = ({ videoUploaded, setAnalysisStarted, uploadedFilePath }) => {
  const handleAnalyze = () => {
    setAnalysisStarted(true);
    axios.post(`http://localhost:3000/analyze_video?file_path=${encodeURIComponent(uploadedFilePath)}`)
      .then(response => {
        console.log("Analysis started:", response.data.message);
      })
      .catch(err => console.error('Error starting video analysis:', err));
  };

  return (
    <button
      type="button"
      className={`flex bg-sky-600 hover:bg-sky-700 text-white px-7 py-3 m-4 rounded-lg w-max cursor-pointer mx-auto shadow-lg ${videoUploaded ? '' : 'opacity-50 cursor-not-allowed'}`}
      onClick={handleAnalyze}
      disabled={!videoUploaded}
    >
      Analyser
    </button>
  );
};
