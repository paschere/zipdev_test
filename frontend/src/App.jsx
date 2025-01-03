import { useState } from "react";

const apiUrl = import.meta.env.VITE_API_URL;

function App() {
  const [jobDescription, setJobDescription] = useState("");
  const [results, setResults] = useState([]);
  const [openDetail, setOpenDetail] = useState(null);
  const [error, setError] = useState(null);

  const highlightSkills = ["ruby", "postgresql", "node", "react", "python"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ job_description: jobDescription }),
      });
      const data = await response.json();
      setResults(data.candidates || []);
      setOpenDetail(null);
      if (data.candidates.length === 0) {
        setError("No candidates found with the given job description");
      }else{
        setError(null);
      }
    } catch (error) {
      setError("An error occurred while fetching data");
      console.error("Error:", error);
    }
  };

  const toggleDetail = (idx) => {
    setOpenDetail(openDetail === idx ? null : idx);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-purple-50 flex flex-col">
      <header className="bg-purple-600 text-white p-4 shadow-lg">
        <h1 className="text-2xl font-bold text-center">
          ZipDev Candidate Matching
        </h1>
        <h2 className="text-sm text-center">
          Mateo Angel
        </h2>
      </header>

      <main className="flex-grow container mx-auto px-4 py-6">
        <div className="max-w-xl mx-auto bg-white rounded-lg shadow p-5 mb-8">
          <h2 className="text-lg font-semibold mb-3 text-gray-700">
            Enter Job Description{" "}
            <span className="text-sm">(max 3500 chars)</span>
          </h2>
          <form onSubmit={handleSubmit}>
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              maxLength={3500}
              rows={4}
              className="w-full border border-gray-300 p-2 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
              placeholder="Describe the role, required skills, etc."
            />
            <button
              type="submit"
              className="mt-4 w-full bg-purple-600 text-white py-2 rounded hover:bg-purple-700 transition-colors"
            >
              Score Candidates
            </button>
          </form>
        </div>

        {error && (
          <div className="max-w-xl mx-auto bg-red-100 text-red-700 p-3 rounded-lg mb-4">
            {error}
          </div>
        )}
        {results.length > 0 && (
          <div className="max-w-3xl mx-auto space-y-6">
            {results.map((cand, idx) => {
              const skillList = cand.Skills
                ? cand.Skills.split("|")
                    .map((s) => s.trim())
                    .filter(Boolean)
                : [];

              return (
                <div
                  key={idx}
                  className={`bg-white shadow rounded-lg p-5 border-l-4 ${idx === 0  ? 'border-green-400' : 'border-purple-400'} relative`}
                >
                  {idx === 0 && ( <div className="absolute top-0 right-0 bg-green-500 text-white px-2 py-1 text-sm font-semibold rounded-bl">
                    Best Match
                  </div>)}

                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-xl font-bold text-gray-800">
                        {cand.Name || "Unnamed Candidate"}
                      </h3>
                      <p className="text-sm text-gray-500">
                        Score: {cand.score?.toFixed(2)}
                      </p>

                      <div className="bg-gray-200 w-32 h-4 rounded-full">
                        <div
                          className="bg-indigo-500 h-4 rounded-full"
                          style={{ width: `${cand.score}%` }}
                        ></div>
                      </div>
                    </div>
                    <button
                      onClick={() => toggleDetail(idx)}
                      className="bg-purple-500 hover:bg-purple-600 text-white px-3 py-1 rounded"
                    >
                      {openDetail === idx ? "Hide Details" : "Show Details"}
                    </button>
                  </div>

                  {openDetail === idx && (
                    <div className="mt-4 text-gray-700">
                      <div className="grid grid-cols-2 gap-4">

                        <div>
                          <p className="font-semibold">Job Title:</p>
                          <p className="mb-2 text-sm text-gray-600">
                            {cand["Job title"] || "No job title"}
                          </p>

                          <p className="font-semibold">Department:</p>
                          <p className="mb-2 text-sm text-gray-600">
                            {cand["Job department"]}
                          </p>

                          <p className="font-semibold">Location:</p>
                          <p className="mb-2 text-sm text-gray-600">
                            {cand["Job location"]}
                          </p>

                          <p className="font-semibold">Stage:</p>
                          <p className="mb-2 text-sm text-gray-600">
                            {cand.Stage}
                          </p>
                        </div>


                        <div>
                          <p className="font-semibold">Skills:</p>
                          {skillList.length > 0 ? (
                            <ul className="flex flex-wrap gap-2 mb-2 mt-1">
                              {skillList.map((skill, sIdx) => {
                                const highlight = highlightSkills.some((h) =>
                                  skill.toLowerCase().includes(h)
                                );
                                return (
                                  <li
                                    key={sIdx}
                                    className={
                                      "px-2 py-1 rounded text-sm " +
                                      (highlight
                                        ? "bg-red-100 text-red-800"
                                        : "bg-gray-200 text-gray-700")
                                    }
                                  >
                                    {skill}
                                  </li>
                                );
                              })}
                            </ul>
                          ) : (
                            <p className="text-sm text-gray-500 mb-2">
                              No Skills
                            </p>
                          )}

                          <p className="font-semibold">Q&A (short):</p>
                          <div className="text-sm text-gray-600 mt-1 space-y-1">
                            {[1, 2, 3, 4, 5, 6, 7].map((n) => {
                              const q = cand[`Question ${n}`];
                              const a = cand[`Answer ${n}`];
                              if (!q && !a) return null;
                              return (
                                <div
                                  key={n}
                                  className="bg-gray-100 p-2 rounded"
                                >
                                  <p className="font-medium text-gray-700">
                                    {q || "Question?"}
                                  </p>
                                  <p className="text-gray-600 italic">
                                    {a || "No answer"}
                                  </p>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>


                      <div className="mt-4">
                        <p className="font-semibold">Experiences:</p>
                        {cand.Experiences ? (
                          <ul className="list-disc list-inside text-sm mt-1">
                            {cand.Experiences.split("|").map((exp, eIdx) => (
                              <li key={eIdx} className="text-gray-600">
                                {exp.trim()}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-sm text-gray-500">
                            No experience data
                          </p>
                        )}
                      </div>

                      <div className="mt-3">
                        <p className="font-semibold">Educations:</p>
                        <p className="text-sm text-gray-600">
                          {cand.Educations || "N/A"}
                        </p>
                      </div>

                      <div className="mt-3 text-sm text-gray-500">
                        <p>Creation time: {cand["Creation time"]}</p>
                        <p>Source: {cand.Source}</p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </main>

      <footer className="bg-gray-200 text-center py-3 text-sm text-gray-600">
        © {new Date().getFullYear()} Candidate Matching Extended
      </footer>
    </div>
  );
}

export default App;
