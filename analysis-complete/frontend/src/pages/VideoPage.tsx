import { useState, useEffect } from 'react';
import { AnalyzeButton } from "../components/ui/analyzeButton";
import { AppBar } from "../components/ui/appBar";
import { HistoryBlock } from "../components/ui/historyBlock";
import { LogBox } from "../components/ui/logBox";
import { UploadButton } from "../components/ui/uploadButton";
import { VideoFeed } from '../components/ui/videoFeed';

export const VideoPage = () => {
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [videoUploaded, setVideoUploaded] = useState<boolean>(false);
  const [uploadedFilePath, setUploadedFilePath] = useState<string>('');
  const [logs, setLogs] = useState<string[]>([]);
  const [vocalLogs, setVocalLogs] = useState<string[]>([]);
  const [analysisStarted, setAnalysisStarted] = useState<boolean>(false);

  useEffect(() => {
    // Call to clear logs on server when component mounts
    fetch('http://localhost:3000/clear_logs', { method: 'POST' })
      .then(response => response.json())
      .then(data => console.log(data.status))
      .catch(err => console.error('Error clearing logs:', err));

    const videoLogInterval = setInterval(() => {
      fetch('http://localhost:3000/logs')
        .then(response => response.text())
        .then(data => {
          setLogs(data.split('\n').filter(line => line));
        });
    }, 100);

    const vocalLogInterval = setInterval(() => {
      fetch('http://localhost:3000/vocal_logs')
          .then(response => response.text())
          .then(data => {
              console.log("Fetched vocal logs:", data);  // Debug print
              setVocalLogs(data.split('\n').filter(line => line));
          });
    }, 500);  // Consider increasing interval for performance


    return () => {
      clearInterval(videoLogInterval);
      clearInterval(vocalLogInterval);
    };
  }, []);

  return (
    <div>
      <AppBar />
      <div className="grid grid-cols-3 mt-8 mx-12">
        <div className="col-span-1 flex justify-center flex-col">
          <UploadButton setUploadProgress={setUploadProgress} setVideoUploaded={setVideoUploaded} setUploadedFilePath={setUploadedFilePath} />
          <AnalyzeButton videoUploaded={videoUploaded} setAnalysisStarted={setAnalysisStarted} uploadedFilePath={uploadedFilePath} />
        </div>
        <div className="col-span-2">
          {videoUploaded && analysisStarted ? (
            <img src={`http://localhost:3000/video_feed?file_path=${encodeURIComponent(uploadedFilePath)}`} alt="Real-time Video Feed" className="h-80 max-w-xl bg-sky-900 rounded-lg" />
          ) : (
            <VideoFeed uploadProgress={uploadProgress} />
          )}
        </div>
      </div>
      <div className="grid grid-cols-11 mx-12 my-5">
        <div className="col-span-6">
          <LogBox label={"Analyse video"} logs={logs} />
          <LogBox label={"Analyse Vocale"} logs={vocalLogs} />
        </div>
        <div className="col-span-5 ml-12 mt-36">
          <HistoryBlock />
        </div>
      </div>
    </div>
  );
};



// import { useState, useEffect } from 'react';

// export const VideoPage = () => {
//     const [logs, setLogs] = useState<string[]>([]);

//     useEffect(() => {
//         const interval = setInterval(() => {
//             fetch('http://localhost:3000/logs')
//                 .then(response => response.text())
//                 .then(data => {
//                     setLogs(data.split('\n').filter(line => line));
//                 });
//         }, 100);

//         return () => clearInterval(interval);
//     }, []);

//     return (
//         <div>
//             <div>
//                 <img src="http://localhost:3000/video_feed" alt="Video Feed" />
//             </div>
//             <div>
//                 <h2>Logs:</h2>
//                 <ul>
//                     {logs.map((log, index) => (
//                         <li key={index}>
//                             <pre>{log}</pre>
//                         </li>
//                     ))}
//                 </ul>
//             </div>
//         </div>
//     );
// };
