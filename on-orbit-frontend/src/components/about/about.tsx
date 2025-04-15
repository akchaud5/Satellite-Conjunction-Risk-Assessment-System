const About = () => {
    return (
        <div className="flex flex-col gap-4 h-full py-16 max-w-[70%] mx-auto">
            <div className="flex flex-col gap-4">
                <h1 className="text-4xl font-medium text-black">About Us</h1>
                <div className="flex flex-col gap-2">
                    <p className="text-lg text-gray-700 mb-4">
                        Welcome to Satellite Conjunction Risk Assessment System. My goal is to provide accurate data on satellites in orbit and predict the probability of collisions to help ensure the safety and sustainability of space operations.
                    </p>
                    <p className="text-lg text-gray-700 mb-4">
                        We leverage advanced orbital mechanics algorithms and machine learning techniques to deliver reliable predictions and insights. Our system uses real-time data to calculate the probability of collision between satellites, helping operators make informed decisions.
                    </p>
                    <p className="text-lg text-gray-700 mb-4">
                        The platform features detailed 3D visualizations of satellite orbits, interactive dashboards for monitoring collision risks, and comprehensive tools for analyzing potential maneuvers to avoid high-risk conjunctions.
                    </p>
                    <p className="text-lg text-gray-700 mb-4">
                        Thank you for visiting our site. If you have any questions or would like to learn more about our project, please feel free to contact us.
                    </p>
                </div>
            </div>
        </div>
    )
}

export default About